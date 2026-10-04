"""ADDENDUM_1.md R2 and R3: N1's matched condition-free control on the seed-42 checks and on the fresh test seeds.
CPU only.

Real run:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
           python src/test/20261108_new_method_quick_checks/matched_controls.py        -> results/addendum1.*
Smoke run: ... matched_controls.py --smoke   -> results/smoke/addendum1.* (reads the smoke outputs of run_checks.py
           and test_seeds.py; E1's smoke seeds 42 and 43 stand in for the test seeds)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_checks as rc  # noqa: E402  (puts run_gonogo and the repo root on sys.path)

from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import METRICS, compare, per_anchor, summarize  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import (ADDENDUM_COMPARATORS, centered_term, config_passes,  # noqa: E402
                                          crossfit_condition_free, decision_row, go_verdict)
from src.eval.aspect_scorers import EvalInputs, agreement_term  # noqa: E402

rg = rc.rg
DEV_SEED = 42
TEST_SEEDS, SMOKE_TEST_SEEDS = (45, 47, 48), (42, 43)
N1_MODELS = ("A3", "C0", "SE")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def terms(inp, ep):
    """(T_u, T_N1, T_N1u): the uniform factor term, the centered term and its condition-removed version."""
    return agreement_term(inp, ep, uniform=True), centered_term(inp, ep), centered_term(inp, ep, uniform=True)


def paired(pa, pb, clusters):
    return {m: compare(pa, pb, clusters, m) for m in ("r1", "gain")}


def dev_part(smoke, res):
    """R2: N1-nested-{A3, C0, SE} on seed 42 against their matched controls; the decision table re-applied."""
    ctx = rg.EvalContext(DEV_SEED, smoke)
    cl = ctx.anchor_group
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    stored = np.load(res / "per_anchor_checks_seed42.npz")
    decision = json.loads((res / "decision.json").read_text())
    models, arrays, n1_configs = {}, {}, {}
    for name in N1_MODELS:
        inp, _, prov = rc.model_inputs(ctx, name, scorer_train, smoke)
        t_u, t_n1, t_n1u = terms(inp, ctx.pooled)
        nested, _, _ = crossfit_nested(ctx.cos, t_u, t_n1, ctx.parity)
        pn = per_anchor(nested)
        if not all(np.array_equal(pn[m], stored[f"{name}__nested_N1__{m}"]) for m in METRICS):
            raise AssertionError(f"{name}: N1-nested differs from run_checks.py's stored arrays")
        matched, picks = crossfit_condition_free(ctx.cos, t_u, t_n1u, ctx.parity)
        pm = per_anchor(matched)
        for m in METRICS:
            arrays[f"dev__{name}__matched__{m}"] = np.asarray(pm[m])
        vs = paired(pn, pm, cl)
        passes = config_passes(vs["r1"], vs["gain"])
        models[name] = {"provenance": prov, "picks": picks, "matched": ctx.summary(pm),
                        "matched_either": rc.point_ci(pm["r1"] + pm["other"], cl),
                        "n1_nested_either": rc.point_ci(pn["r1"] + pn["other"], cl),
                        "vs_matched": vs, "passes": passes}
        n1_configs[f"N1-nested-{name}"] = {"name": f"N1-nested-{name}", "passes": passes, "m_r": vs["r1"]["point"],
                                           "m_g": vs["gain"]["point"], "r1_ci95": vs["r1"]["ci95"],
                                           "gain_ci95": vs["gain"]["ci95"]}
        log(f"dev {name} done")
    configs = [n1_configs.get(c["name"], c) for c in decision["configs"]]
    return {"models": models, "configs": configs, "d0_reading": decision["d0"]["reading"],
            "decision": decision_row(configs, decision["d0"]["reading"]),
            "original_decision": decision["decision"]}, arrays


def test_part(smoke, res):
    """R3: A3's matched control on each test seed, pooled with test_seeds.py's stored arrays; GO with four comparators."""
    stored = np.load(res / "per_anchor_test_seeds.npz")
    test = json.loads((res / "test_seeds.json").read_text())
    seeds = SMOKE_TEST_SEEDS if smoke else TEST_SEEDS
    if test["seeds"] != list(seeds):
        raise AssertionError(f"test_seeds.json holds seeds {test['seeds']}, expected {list(seeds)}")
    scorers = ("config", *ADDENDUM_COMPARATORS)
    pooled = {s: {m: [] for m in METRICS} for s in scorers}
    clusters, per_seed, picks_all, arrays = [], {}, {}, {}
    for seed in seeds:
        ctx = rg.EvalContext(seed, smoke)
        cl = ctx.anchor_group
        if not np.array_equal(stored[f"seed{seed}__anchor_group"], cl):
            raise AssertionError(f"seed {seed}: anchor groups differ from test_seeds.py's stored arrays")
        ckpt = rg.checkpoint_path("A3", 42, smoke)
        if not smoke and rg.sha_file(ckpt) != rc.A3_SHA:
            raise AssertionError(f"{ckpt}: not A3's checkpoint")
        ic, tc = ctx.encode(ckpt)
        t_u, t_n1, t_n1u = terms(EvalInputs(ctx.img, ctx.txt, ic, tc), ctx.pooled)
        nested, _, _ = crossfit_nested(ctx.cos, t_u, t_n1, ctx.parity)
        pa = {"config": per_anchor(nested)}
        if not all(np.array_equal(pa["config"][m], stored[f"seed{seed}__config__{m}"]) for m in METRICS):
            raise AssertionError(f"seed {seed}: N1-nested differs from test_seeds.py's stored arrays")
        matched, picks = crossfit_condition_free(ctx.cos, t_u, t_n1u, ctx.parity)
        pa["matched"] = per_anchor(matched)
        for s in ("cosine", "rca", "control"):
            pa[s] = {m: stored[f"seed{seed}__{s}__{m}"].astype(np.float64) for m in METRICS}
        per_seed[str(seed)] = {"matched": ctx.summary(pa["matched"]),
                               "matched_either": rc.point_ci(pa["matched"]["r1"] + pa["matched"]["other"], cl),
                               "vs": {c: paired(pa["config"], pa[c], cl) for c in ADDENDUM_COMPARATORS}}
        picks_all[str(seed)] = picks
        for m in METRICS:
            arrays[f"test__seed{seed}__matched__{m}"] = np.asarray(pa["matched"][m])
        for s in scorers:
            for m in METRICS:
                pooled[s][m].append(np.asarray(pa[s][m], dtype=np.float64))
        clusters.append(cl)
        log(f"test seed {seed} done")
    P = {s: {m: np.concatenate(v) for m, v in d.items()} for s, d in pooled.items()}
    cl = np.concatenate(clusters)
    vs = {c: paired(P["config"], P[c], cl) for c in ADDENDUM_COMPARATORS}
    for c in ("cosine", "rca", "control"):
        for m in ("r1", "gain"):
            if vs[c][m]["point"] != test["pooled"]["vs"][c][m]["point"] or \
                    vs[c][m]["ci95"] != test["pooled"]["vs"][c][m]["ci95"]:
                raise AssertionError(f"pooled config − {c} {m} differs from test_seeds.json")
    return {"per_seed": per_seed, "picks": picks_all,
            "pooled": {"vs": vs, "matched": summarize(P["matched"], cl),
                       "matched_either": rc.point_ci(P["matched"]["r1"] + P["matched"]["other"], cl)},
            "go_addendum": go_verdict(vs, ADDENDUM_COMPARATORS), "go_section6": test["go"]}, arrays


def summary_text(r):
    def c(x):
        return f"{x['point']:6.2f} [{x['ci95'][0]:6.2f}, {x['ci95'][1]:6.2f}]"

    lines = [f"ADDENDUM_1 matched controls{' (SMOKE)' if r['smoke'] else ''}", "R2, seed 42:"]
    for name, e in r["dev"]["models"].items():
        lines.append(f"  {name}: matched control R@1 {c(e['matched']['r1'])} either {e['matched_either']['point']:6.2f} "
                     f"picks {e['picks']}; N1-nested − matched R@1 {c(e['vs_matched']['r1'])} gain "
                     f"{c(e['vs_matched']['gain'])} pass {e['passes']}")
    d = r["dev"]["decision"]
    lines.append(f"  re-applied table: row {d['row']} ({d['config']}); original: row {r['dev']['original_decision']['row']}")
    lines.append("R3, test seeds:")
    for seed, e in r["test"]["per_seed"].items():
        lines.append(f"  seed {seed}: matched R@1 {c(e['matched']['r1'])}; config − matched R@1 {c(e['vs']['matched']['r1'])} "
                     f"gain {c(e['vs']['matched']['gain'])}; picks {r['test']['picks'][seed]}")
    p = r["test"]["pooled"]
    lines.append(f"  pooled matched R@1 {c(p['matched']['r1'])} either {p['matched_either']['point']:6.2f}")
    for comp in ADDENDUM_COMPARATORS:
        lines.append(f"  pooled config − {comp:7s} R@1 {c(p['vs'][comp]['r1'])} gain {c(p['vs'][comp]['gain'])}")
    lines.append(f"  GO (addendum, governs): {r['test']['go_addendum']}; GO (§6 as committed): {r['test']['go_section6']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    res = HERE / "results" / ("smoke" if smoke else "")
    names = ("addendum1.json", "addendum1.txt", "per_anchor_addendum1.npz")
    if not smoke and any((res / n).exists() for n in names):
        raise SystemExit(f"results exist in {res}; refusing to overwrite")
    t0 = time.time()
    dev, dev_arrays = dev_part(smoke, res)
    test, test_arrays = test_part(smoke, res)
    result = {"smoke": smoke, "dev": dev, "test": test,
              "provenance": {"addendum_sha256": rg.sha_file(HERE / "ADDENDUM_1.md"),
                             "script_sha256": rg.sha_file(Path(__file__)),
                             "module_sha256": rg.sha_file(rc.ROOT / "src/eval/aspect_quick_checks.py")}}
    rg.assert_finite_tree(result)
    (res / "addendum1.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(res / "per_anchor_addendum1.npz", **dev_arrays, **test_arrays)
    text = summary_text(result)
    (res / "addendum1.txt").write_text(text + "\n")
    print(text)
    log(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
