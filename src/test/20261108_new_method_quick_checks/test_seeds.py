"""Fresh-seed test of N1-nested-A3 (TEST_CONFIG.md in this folder; GO rule DECISION_RULE.md §6). CPU only.

Real run:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
           python src/test/20261108_new_method_quick_checks/test_seeds.py         -> results/ (refuses to overwrite)
Smoke run: ... test_seeds.py --smoke  -> results/smoke/ (E1's smoke seed-42 and seed-43 episodes stand in for the test
           seeds; A1's smoke checkpoint stands in for A3)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src/test/20261101_aspect_factor_gonogo"))
import run_gonogo as rg  # noqa: E402  (puts the repo root on sys.path)

from src.eval.aspect_metrics import METRICS, cluster_bootstrap, compare, per_anchor, summarize  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import GO_COMPARATORS, centered_term, go_verdict  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, agreement_term  # noqa: E402

SEEDS, SMOKE_SEEDS = (45, 47, 48), (42, 43)
A3_SHA = "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2"
SCORERS = ("config", "control", "cosine", "rca")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def point_ci(values, clusters):
    r = cluster_bootstrap(values, clusters)
    return {"point": 100 * r["point"], "ci95": [100 * c for c in r["ci95"]]}


def assert_finite_scores(scores, what):
    for c in scores:
        for d in scores[c]:
            bad = int((~np.isfinite(np.asarray(scores[c][d]))).any(axis=1).sum())
            if bad:
                raise AssertionError(f"{what}: {bad} episodes with a non-finite score ({c}, {d})")


def score_seed(seed, smoke):
    """Per-anchor arrays of the configuration, its control, cosine and RCA on one seed, plus provenance."""
    ctx = rg.EvalContext(seed, smoke)
    t9 = np.load(rg.folders(smoke)["e1"] / f"per_anchor_seed{seed}.npz")
    cos_pa = per_anchor(ctx.cos)
    if not (all(np.array_equal(cos_pa[m], t9[f"cosine__{m}"]) for m in METRICS)
            and np.array_equal(t9["anchor_group"], ctx.anchor_group)
            and np.array_equal(t9["pair_index"], ctx.pair_index)):
        raise AssertionError(f"seed {seed}: episodes misaligned with E1's per-anchor arrays")
    ckpt = rg.checkpoint_path("A3", 42, smoke)                       # smoke: A1's smoke checkpoint stands in
    sha = rg.sha_file(ckpt)
    if not smoke and sha != A3_SHA:
        raise AssertionError(f"{ckpt}: SHA-256 {sha} is not A3's (TEST_CONFIG.md)")
    ic, tc = ctx.encode(ckpt)
    inp = EvalInputs(ctx.img, ctx.txt, ic, tc)
    t_u, t_n1 = agreement_term(inp, ctx.pooled, uniform=True), centered_term(inp, ctx.pooled)
    for name, s in (("T_u", t_u), ("T_N1", t_n1), ("cosine", ctx.cos)):
        assert_finite_scores(s, f"seed {seed} {name}")
    nested, control, picks = crossfit_nested(ctx.cos, t_u, t_n1, ctx.parity)
    assert_finite_scores(nested, f"seed {seed} nested")
    pa = {"config": per_anchor(nested), "control": per_anchor(control), "cosine": cos_pa,
          "rca": {m: t9[f"rca__{m}"].astype(np.float64) for m in METRICS}}
    prov = {"episodes_sha256": ctx.shas, "n_episodes": int(ctx.n), "checkpoint": str(ckpt.relative_to(ROOT)),
            "checkpoint_sha256": sha, "per_anchor_e1_sha256": rg.sha_file(rg.folders(smoke)["e1"] /
                                                                          f"per_anchor_seed{seed}.npz")}
    return pa, ctx.anchor_group, ctx.pair_index, picks, prov


def block(pa, clusters, pair_index):
    """Summaries of every scorer, either rates, paired differences config − comparator, and per aspect pair."""
    out = {s: {"summary": summarize({m: pa[s][m] for m in METRICS}, clusters),
               "either": point_ci(pa[s]["r1"] + pa[s]["other"], clusters)} for s in SCORERS}
    out["vs"] = {c: {m: compare(pa["config"], pa[c], clusters, m) for m in ("r1", "gain")} for c in GO_COMPARATORS}
    out["per_pair"] = {}
    for i, name in enumerate(rg.POOLED_ORDER):
        mask = pair_index == i
        sub = {s: {m: pa[s][m][mask] for m in METRICS} for s in SCORERS}
        out["per_pair"][name] = {
            "config": summarize(sub["config"], clusters[mask]),
            "control": summarize(sub["control"], clusters[mask]),
            "vs": {c: {m: compare(sub["config"], sub[c], clusters[mask], m) for m in ("r1", "gain")}
                   for c in GO_COMPARATORS}}
    return out


def summary_text(r):
    def c(x):
        return f"{x['point']:6.2f} [{x['ci95'][0]:6.2f}, {x['ci95'][1]:6.2f}]"

    lines = [f"Fresh-seed test of N1-nested-A3, seeds {r['seeds']}{' (SMOKE)' if r['smoke'] else ''}"]
    for key in [*(str(s) for s in r["seeds"]), "pooled"]:
        b = r["per_seed"][key] if key != "pooled" else r["pooled"]
        lines.append(f"{key}:")
        for s in SCORERS:
            lines.append(f"  {s:7s} R@1 {c(b[s]['summary']['r1'])} gain {c(b[s]['summary']['gain'])} "
                         f"either {b[s]['either']['point']:6.2f}")
        for comp in GO_COMPARATORS:
            lines.append(f"  config - {comp:7s} R@1 {c(b['vs'][comp]['r1'])} gain {c(b['vs'][comp]['gain'])}")
        if key != "pooled":
            lines.append(f"  picks {r['picks'][key]}")
    lines.append(f"GO: {r['go']['go']} (failed: {r['go']['failed']})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    res = HERE / "results" / ("smoke" if smoke else "")
    names = ("test_seeds.json", "per_anchor_test_seeds.npz", "test_seeds.txt")
    if not smoke and any((res / n).exists() for n in names):
        raise SystemExit(f"results exist in {res}; the test runs once. Refusing to overwrite.")
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = SMOKE_SEEDS if smoke else SEEDS
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    result = {"seeds": list(seeds), "smoke": smoke, "per_seed": {}, "picks": {},
              "provenance": {"git_head": head, "script_sha256": rg.sha_file(Path(__file__)),
                             "test_config_sha256": rg.sha_file(HERE / "TEST_CONFIG.md"),
                             "decision_rule_sha256": rg.sha_file(HERE / "DECISION_RULE.md"),
                             "module_sha256": {m: rg.sha_file(ROOT / f"src/eval/{m}.py") for m in
                                               ("aspect_quick_checks", "aspect_nested", "aspect_scorers",
                                                "aspect_metrics")},
                             "seeds": {}}}
    arrays, pooled_pa, pooled_cl, pooled_pi = {}, {s: {m: [] for m in METRICS} for s in SCORERS}, [], []
    for seed in seeds:
        pa, clusters, pair_index, picks, prov = score_seed(seed, smoke)
        result["per_seed"][str(seed)] = block(pa, clusters, pair_index)
        result["picks"][str(seed)] = picks
        result["provenance"]["seeds"][str(seed)] = prov
        arrays[f"seed{seed}__anchor_group"], arrays[f"seed{seed}__pair_index"] = clusters, pair_index
        for s in SCORERS:
            for m in METRICS:
                arrays[f"seed{seed}__{s}__{m}"] = np.asarray(pa[s][m])
                pooled_pa[s][m].append(np.asarray(pa[s][m], dtype=np.float64))
        pooled_cl.append(clusters)
        pooled_pi.append(pair_index)
        log(f"seed {seed} done")
    pooled = {s: {m: np.concatenate(v) for m, v in d.items()} for s, d in pooled_pa.items()}
    clusters, pair_index = np.concatenate(pooled_cl), np.concatenate(pooled_pi)
    result["pooled"] = block(pooled, clusters, pair_index)
    result["pooled"]["n_episodes"], result["pooled"]["n_clusters"] = int(len(clusters)), int(len(np.unique(clusters)))
    result["go"] = go_verdict(result["pooled"]["vs"])
    rg.assert_finite_tree(result)
    (res / "test_seeds.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(res / "per_anchor_test_seeds.npz", **arrays)
    text = summary_text(result)
    (res / "test_seeds.txt").write_text(text + "\n")
    print(text)
    log(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
