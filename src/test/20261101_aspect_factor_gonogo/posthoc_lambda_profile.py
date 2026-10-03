"""POST-HOC, DESCRIPTIVE checks (not pre-registered; outside the E3 decision map; written after the final review of
2026-10-03). They change no pick, rule or verdict of the go/no-go. CPU only, selection rows only, stored inputs only.

1. Fixed-lambda profile (no cross-fitting). On the seed-42 and seed-43 selection episodes, for A3 (the picked run),
   C0 and SE (Task 9's cached codes) and their uniform-weight controls: R@1, condition gain and the "either aspect
   candidate first" rate (R@1 + other-aspect rate) of z(cos) + lam * z(term) at every lam of the fusion grid, with
   painting-clustered bootstrap intervals, plus the paired term-only (lam = inf) gains A3 - SE and A3 - C0.
2. Supervision ablation on both episode draws: S1 - A1 on the emotion pairs, seed 43 from per_anchor_gonogo_seed43.npz
   and seed 42 from per_anchor_select_seed42.npz (the stored cross-fitted per-anchor arrays of both runs).
3. Bootstrap-seed sensitivity of the GO comparisons: the six paired lower bounds of the GO test (A3 against backbone
   only, the GO bar RCA and its uniform-weight control, on R@1 and condition gain) recomputed with bootstrap seeds
   0..99, 5,000 resamples each. Seed 42 is the pre-registered one and must reproduce gonogo.json exactly.

Run from the repository root:
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python \
      src/test/20261101_aspect_factor_gonogo/posthoc_lambda_profile.py
Writes results/posthoc_lambda_profile.json (summaries and counts), results/posthoc_lambda_profile_seed{42,43}.npz
(per-anchor arrays, so build_figures.py can re-derive every summary) and results/posthoc_lambda_profile.txt.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_gonogo as rg  # noqa: E402  (puts the repo root on sys.path)

from src.eval.aspect_metrics import METRICS, cluster_bootstrap, compare, per_anchor  # noqa: E402
from src.eval.aspect_scorers import LAMBDA_GRID, EvalInputs, agreement_term, fused_scores  # noqa: E402

NOTE = ("POST-HOC, DESCRIPTIVE; not pre-registered; outside the E3 decision map; fixed lambda (no cross-fitting) in "
        "the profile; changes no pick, rule or verdict")
MODELS = ("A3", "C0", "SE")
EMOTION_PAIRS = rg.EMOTION_PAIRS
MC_SEEDS = range(100)
RES = HERE / "results"


def lam_key(lam: float) -> str:
    return "inf" if np.isinf(lam) else f"{lam:g}"


def pts(result: dict) -> dict:
    return {"point": 100 * result["point"], "ci95": [100 * c for c in result["ci95"]], "n_clusters": result["n_clusters"]}


def boot(values, clusters, seed: int = 42) -> dict:
    return pts(cluster_bootstrap(values, clusters, seed=seed))


def codes_for(ctx, name: str):
    if name == "A3":
        ckpt = rg.checkpoint_path("A3", 42, False)
        pick = json.loads((RES / "picked.json").read_text())
        assert pick["run"] == "A3" and rg.sha_file(ckpt) == pick["checkpoint_sha256"], "A3 checkpoint != the pick"
        return ctx.encode(ckpt), {"checkpoint": str(ckpt.relative_to(rg.ROOT)), "sha256": rg.sha_file(ckpt)}
    path = rg.E1 / f"codes_{name}.npz"
    z = np.load(path)
    return (ctx.masked(z["img"]), ctx.masked(z["txt"])), {"codes": str(path.relative_to(rg.ROOT)),
                                                           "sha256": rg.sha_file(path)}


def lambda_profile(seed: int, out: dict, arrays: dict) -> None:
    ctx = rg.EvalContext(seed, smoke=False)
    cl = ctx.anchor_group
    stored_runs = np.load(RES / ("per_anchor_gonogo_seed43.npz" if seed == 43 else "per_anchor_select_seed42.npz"))
    t9 = np.load(rg.E1 / f"per_anchor_seed{seed}.npz")
    cos_pa = per_anchor(ctx.cos)
    assert all(np.array_equal(cos_pa[m], t9[f"cosine__{m}"]) for m in METRICS), "episodes misaligned with Task 9"
    entry = {"episodes_sha256": ctx.shas, "n_episodes": ctx.n, "n_clusters": int(len(np.unique(cl))),
             "cosine": {"r1": boot(cos_pa["r1"], cl), "either": boot(cos_pa["r1"] + cos_pa["other"], cl)},
             "models": {}, "inputs": {}}
    term_only = {}
    for name in MODELS:
        (ic, tc), prov = codes_for(ctx, name)
        entry["inputs"][name] = prov
        inp = EvalInputs(ctx.img, ctx.txt, ic, tc)
        # alignment check: the cross-fitted scores recomputed here equal the stored per-anchor arrays
        pa_cf, _ = ctx.score(ic, tc, uniform=False)
        ref = (stored_runs, "picked" if seed == 43 else "A3") if name == "A3" else (t9, name)
        assert all(np.array_equal(pa_cf[m], ref[0][f"{ref[1]}__{m}"]) for m in METRICS), f"{name} seed {seed}"
        entry["models"][name] = {}
        for weights, uniform in (("agreement", False), ("uniform", True)):
            term = agreement_term(inp, ctx.pooled, uniform=uniform)
            prof = {}
            for lam in LAMBDA_GRID:
                pa = per_anchor(fused_scores(ctx.cos, term, lam))
                either = pa["r1"] + pa["other"]
                prof[lam_key(lam)] = {"r1": boot(pa["r1"], cl), "gain": boot(pa["gain"], cl),
                                      "other": 100 * float(pa["other"].mean()), "either": boot(either, cl)}
                for m in ("r1", "gain", "other"):
                    arrays[f"{name}__{weights}__{lam_key(lam)}__{m}"] = pa[m]
                if weights == "agreement" and np.isinf(lam):
                    term_only[name] = pa
            entry["models"][name][weights] = prof
            print(f"seed {seed} {name} {weights}: " + "  ".join(
                f"{k} {v['r1']['point']:.2f}/{v['gain']['point']:+.2f}/{v['either']['point']:.1f}"
                for k, v in prof.items()), flush=True)
    entry["term_only_paired_gain"] = {f"A3_minus_{o}": compare(term_only["A3"], term_only[o], cl, "gain")
                                      for o in ("SE", "C0")}
    entry["term_only_paired_r1"] = {f"A3_minus_{o}": compare(term_only["A3"], term_only[o], cl, "r1")
                                    for o in ("SE", "C0")}
    arrays["anchor_group"], arrays["pair_index"] = cl, ctx.pair_index
    out[f"seed{seed}"] = entry


def ablation(out: dict) -> None:
    gj = json.loads((RES / "gonogo.json").read_text())
    res = {"note": "S1 (IC bank) minus A1 (AIC bank) on the emotion pairs; one model seed (42) per arm, so the "
                   "intervals carry episode and painting variance but no training variance"}
    for seed, fname in ((43, "per_anchor_gonogo_seed43.npz"), (42, "per_anchor_select_seed42.npz")):
        z = np.load(RES / fname)
        cl, emo = z["anchor_group"], np.isin(z["pair_index"], EMOTION_PAIRS)
        pa = {r: {m: z[f"{r}__{m}"].astype(np.float64)[emo] for m in METRICS} for r in ("A1", "S1")}
        entry = {"n_episodes": int(emo.sum()), "n_clusters": int(len(np.unique(cl[emo])))}
        for m in ("r1", "gain"):
            entry[f"S1_minus_A1_{m}"] = compare(pa["S1"], pa["A1"], cl[emo], m)
            for r in ("A1", "S1"):
                entry[f"{r}_{m}"] = boot(pa[r][m], cl[emo])
        res[f"seed{seed}"] = entry
    for m in ("r1", "gain"):
        stored = gj["ablation"]["S1_vs_A1_emotion_pairs"][m]
        got = res["seed43"][f"S1_minus_A1_{m}"]
        assert abs(got["point"] - stored["point"]) < 1e-9 and np.allclose(got["ci95"], stored["ci95"], atol=1e-9, rtol=0)
    out["ablation_emotion_pairs"] = res


def bootstrap_seed_sensitivity(out: dict) -> None:
    gj = json.loads((RES / "gonogo.json").read_text())
    z = np.load(RES / "per_anchor_gonogo_seed43.npz")
    t9 = np.load(rg.E1 / "per_anchor_seed43.npz")
    assert np.array_equal(z["anchor_group"], t9["anchor_group"])
    cl = z["anchor_group"]
    pk = {m: z[f"picked__{m}"] for m in ("r1", "gain")}
    others = {"backbone_only": {m: t9[f"cosine__{m}"] for m in ("r1", "gain")},
              "go_baseline": {m: t9[f"rca__{m}"] for m in ("r1", "gain")},
              "uniform_control": {m: z[f"picked_uniform__{m}"] for m in ("r1", "gain")}}
    lbs = {f"{k}__{m}": np.array([100 * cluster_bootstrap(pk[m] - o[m], cl, seed=s)["ci95"][0] for s in MC_SEEDS])
           for k, o in others.items() for m in ("r1", "gain")}
    for k in others:
        for m in ("r1", "gain"):
            assert abs(lbs[f"{k}__{m}"][42] - gj["go"]["comparators"][k][m]["ci95"][0]) < 1e-9, (k, m)
    passes = {k: v > 0 for k, v in lbs.items()}
    beats = {k: passes[f"{k}__r1"] & passes[f"{k}__gain"] for k in others}
    go = beats["backbone_only"] & beats["go_baseline"] & beats["uniform_control"]
    out["bootstrap_seed_sensitivity"] = {
        "note": "seed-43 GO comparisons with bootstrap seeds 0..99 (5,000 resamples each); seed 42 is the "
                "pre-registered bootstrap and reproduces gonogo.json exactly",
        "n_seeds": len(MC_SEEDS),
        "lower_bound_above_0": {k: int(v.sum()) for k, v in passes.items()},
        "lower_bound_median": {k: float(np.median(v)) for k, v in lbs.items()},
        "lower_bound_range": {k: [float(v.min()), float(v.max())] for k, v in lbs.items()},
        "beats_both_metrics": {k: int(v.sum()) for k, v in beats.items()},
        "GO_all_six": int(go.sum()),
        "seed42_lower_bounds": {k: float(v[42]) for k, v in lbs.items()}}


def main() -> None:
    RES.mkdir(exist_ok=True)
    out = {"note": NOTE, "lambda_grid": [lam_key(x) for x in LAMBDA_GRID], "bootstrap": "5,000 resamples, seed 42, "
           "painting clusters (section 3 varies the seed)", "script_sha256": rg.sha_file(Path(__file__))}
    for seed in (43, 42):
        arrays: dict = {}
        lambda_profile(seed, out, arrays)
        np.savez(RES / f"posthoc_lambda_profile_seed{seed}.npz", **arrays)
    ablation(out)
    bootstrap_seed_sensitivity(out)
    rg.assert_finite_tree(out)
    (RES / "posthoc_lambda_profile.json").write_text(json.dumps(out, indent=1))

    def c(x):
        return f"{x['point']:+6.2f} [{x['ci95'][0]:+6.2f},{x['ci95'][1]:+6.2f}]"

    lines = [NOTE]
    for seed in (43, 42):
        e = out[f"seed{seed}"]
        lines.append(f"\n== seed {seed} episodes ({e['n_episodes']} pooled, {e['n_clusters']} paintings); cosine R@1 "
                     f"{e['cosine']['r1']['point']:.2f}, either {e['cosine']['either']['point']:.2f}")
        for name in MODELS:
            for weights in ("agreement", "uniform"):
                lines.append(f"  {name} {weights}: lambda   R@1                      gain                     either")
                for k, v in e["models"][name][weights].items():
                    lines.append(f"    {k:>5s}  {c(v['r1'])}  {c(v['gain'])}  {c(v['either'])}")
        for o in ("SE", "C0"):
            lines.append(f"  term-only (lambda inf) paired gain A3 - {o}: {c(e['term_only_paired_gain'][f'A3_minus_{o}'])}"
                         f"   R@1 {c(e['term_only_paired_r1'][f'A3_minus_{o}'])}")
    a = out["ablation_emotion_pairs"]
    lines.append("\n== supervision ablation, emotion pairs (one model seed per arm)")
    for seed in (43, 42):
        s = a[f"seed{seed}"]
        lines.append(f"  episodes seed {seed}: S1 - A1 gain {c(s['S1_minus_A1_gain'])}, R@1 {c(s['S1_minus_A1_r1'])}; "
                     f"A1 gain {c(s['A1_gain'])}; S1 gain {c(s['S1_gain'])}")
    b = out["bootstrap_seed_sensitivity"]
    lines.append(f"\n== bootstrap-seed sensitivity of the GO comparisons ({b['n_seeds']} seeds)")
    for k, v in b["lower_bound_above_0"].items():
        lines.append(f"  {k:24s} lower bound > 0 in {v:3d}/{b['n_seeds']}; median {b['lower_bound_median'][k]:+.4f}, "
                     f"range [{b['lower_bound_range'][k][0]:+.4f}, {b['lower_bound_range'][k][1]:+.4f}]")
    lines.append(f"  beats on both metrics: {b['beats_both_metrics']}; GO (all six) in {b['GO_all_six']}/{b['n_seeds']}")
    text = "\n".join(lines)
    print(text)
    (RES / "posthoc_lambda_profile.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
