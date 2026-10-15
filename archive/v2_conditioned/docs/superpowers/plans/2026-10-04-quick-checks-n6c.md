# N6c gate and test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ADDENDUM_3_N6C.md`: score N6c (N6's hard-reader term on A3's centered uniform factor base) against cosine, RCA, its nested control C1 and its matched control C2, as a seed-42 gate and, if the gate passes, a pooled test on seeds 45, 47, 48.

**Architecture:** One runner `src/test/20261108_new_method_quick_checks/run_n6c.py` with `--stage gate|test`, built only from reviewed pieces: `run_checks.py` helpers, `run_n6.py` (`load_posteriors`, `n6_terms`, `e1_arrays`), `centered_term`, `crossfit_nested`, `crossfit_condition_free`, `go_verdict`.

**Tech Stack:** Python 3.10 (conda env `CoSiR`), numpy.

**Spec:** `src/test/20261108_new_method_quick_checks/ADDENDUM_3_N6C.md` (8ff2fc5). Read it first.

## Global Constraints

- Python: `/root/miniconda3/envs/CoSiR/bin/python`; never install anything. CPU only: prefix every command with `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`.
- Implementers run only `--smoke` (both stages). Never run any runner of the folder without `--smoke`; never open non-smoke files under `src/test/20261108_new_method_quick_checks/results/`.
- Git: `main`; stage by explicit path; commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01VzPGkSgZ3UrCEi5JYqTDw1`.
- Create only `run_n6c.py`; edit no existing file.
- Gate: R@1 and gain against C1 and C2 all with 95% lower bounds above 0 on seed 42. GO: pooled over seeds 45, 47, 48, R@1 and gain against cosine, RCA, C1 and C2 all with lower bounds above 0 (painting clusters, 5,000 resamples, seed 42).

## Review Focus

1. **The matched control must hold T_6u** (N6c with the condition removed), not T_6: `crossfit_condition_free` raises on a conditioned term.
2. **Frozen heads:** the runner loads `n6_posteriors.npz` and checks its SHA-256 (real run); it never refits heads.
3. **Gate before test:** the real test stage exits unless `n6c_gate.json` says `passes: true`.
4. **Comparator direction and pooling** follow `test_seeds.py` and `matched_controls.py` (config − comparator; one cluster per painting across seeds).
5. **Overwrite:** real outputs are write-once.

---

### Task 1: The N6c runner

**Files:**
- Create: `src/test/20261108_new_method_quick_checks/run_n6c.py`

**Interfaces:**
- Consumes: `run_checks.py` as `rc` (`rc.rg`, `rc.A3_SHA`, `rc.point_ci`, `rc.assert_finite_scores`, `rc.log`, `rc.ROOT`); `run_n6.py` as `n6` (`n6.load_posteriors(path, ctx)`, `n6.n6_terms(post, ep) -> (T6, T6soft, T6u, info, soft_info)`, `n6.e1_arrays(ctx, smoke, seed) -> (cos_pa, rca_pa)`); `src.eval.aspect_quick_checks`: `ADDENDUM_COMPARATORS = ("cosine", "rca", "control", "matched")`, `centered_term`, `crossfit_condition_free(cos, t_u, t_c, parity) -> (scores, picks)`, `go_verdict(pooled, comparators)`; `src.eval.aspect_nested.crossfit_nested(cos, t_u, t_a, parity) -> (nested, control, picks)`; `src.eval.aspect_scorers.EvalInputs`; `src.eval.aspect_metrics.METRICS, compare, per_anchor, summarize`.
- Produces: `results/n6c_gate.{json,txt}`, `results/per_anchor_n6c_gate.npz`; `results/n6c_test.{json,txt}`, `results/per_anchor_n6c_test.npz` (smoke: `results/smoke/`).

- [ ] **Step 1: Write the runner**

Create `src/test/20261108_new_method_quick_checks/run_n6c.py`:

```python
"""N6c (ADDENDUM_3_N6C.md in this folder): N6's hard-reader term on A3's centered uniform factor base, against its
nested control C1 and its matched control C2; a seed-42 gate, then the fresh-seed test. CPU only.

Gate:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
       python src/test/20261108_new_method_quick_checks/run_n6c.py --stage gate    -> results/n6c_gate.*
Test:  ... run_n6c.py --stage test    -> results/n6c_test.* (exits unless the gate passed)
Smoke: add --smoke (results/smoke/; smoke posteriors and episodes; seeds 42 and 43 stand in for the test seeds; the
       smoke test stage runs whatever the smoke gate found)
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
import run_n6 as n6  # noqa: E402

from src.eval.aspect_metrics import METRICS, compare, per_anchor, summarize  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import (ADDENDUM_COMPARATORS, centered_term, crossfit_condition_free,  # noqa: E402
                                          go_verdict)
from src.eval.aspect_scorers import EvalInputs  # noqa: E402

rg = rc.rg
POSTERIORS_SHA = "2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0"
GATE_SEED, TEST_SEEDS, SMOKE_TEST_SEEDS = 42, (45, 47, 48), (42, 43)
SCORERS = ("config", *ADDENDUM_COMPARATORS)        # config, cosine, rca, control (C1), matched (C2)


def score_seed(seed, smoke, res):
    """Per-anchor arrays of N6c, C1, C2, cosine and RCA on one episode seed, with picks and provenance."""
    ctx = rg.EvalContext(seed, smoke)
    post_path = res / "n6_posteriors.npz"
    if not smoke and rg.sha_file(post_path) != POSTERIORS_SHA:
        raise AssertionError(f"{post_path}: SHA-256 differs from ADDENDUM_3_N6C.md")
    post = n6.load_posteriors(post_path, ctx)
    ckpt = rg.checkpoint_path("A3", 42, smoke)                       # smoke: A1's smoke checkpoint stands in
    sha = rg.sha_file(ckpt)
    if not smoke and sha != rc.A3_SHA:
        raise AssertionError(f"{ckpt}: not A3's checkpoint")
    ic, tc = ctx.encode(ckpt)
    t_n1u = centered_term(EvalInputs(ctx.img, ctx.txt, ic, tc), ctx.pooled, uniform=True)
    t6, _, t6u, _, _ = n6.n6_terms(post, ctx.pooled)
    for name, s in (("T_N1u", t_n1u), ("T_6", t6), ("T_6u", t6u)):
        rc.assert_finite_scores(s, f"seed {seed} {name}")
    nested, c1, picks = crossfit_nested(ctx.cos, t_n1u, t6, ctx.parity)
    c2, picks_c2 = crossfit_condition_free(ctx.cos, t_n1u, t6u, ctx.parity)
    cos_pa, rca_pa = n6.e1_arrays(ctx, smoke, seed)
    pa = {"config": per_anchor(nested), "control": per_anchor(c1), "matched": per_anchor(c2),
          "cosine": cos_pa, "rca": rca_pa}
    prov = {"checkpoint_sha256": sha, "episodes_sha256": ctx.shas, "posteriors_sha256": rg.sha_file(post_path)}
    return ctx, pa, {"nested": picks, "matched": picks_c2}, prov


def block(pa, clusters, pair_index=None):
    out = {s: {"summary": summarize(pa[s], clusters), "either": rc.point_ci(pa[s]["r1"] + pa[s]["other"], clusters)}
           for s in SCORERS}
    out["vs"] = {c: {m: compare(pa["config"], pa[c], clusters, m) for m in ("r1", "gain")} for c in ADDENDUM_COMPARATORS}
    if pair_index is not None:
        out["per_pair_vs"] = {}
        for i, name in enumerate(rg.POOLED_ORDER):
            mask = pair_index == i
            sub = {s: {m: pa[s][m][mask] for m in METRICS} for s in ("config", "control", "matched")}
            out["per_pair_vs"][name] = {c: {m: compare(sub["config"], sub[c], clusters[mask], m) for m in ("r1", "gain")}
                                        for c in ("control", "matched")}
    return out


def gate_stage(smoke, res):
    ctx, pa, picks, prov = score_seed(GATE_SEED, smoke, res)
    b = block(pa, ctx.anchor_group, ctx.pair_index)
    passes = all(b["vs"][c][m]["ci95"][0] > 0 for c in ("control", "matched") for m in ("r1", "gain"))
    arrays = {"anchor_group": ctx.anchor_group, "pair_index": ctx.pair_index,
              **{f"{s}__{m}": np.asarray(pa[s][m]) for s in SCORERS for m in METRICS}}
    return {"seed": GATE_SEED, "picks": picks, "provenance": prov, **b, "passes": bool(passes)}, arrays


def test_stage(smoke, res):
    gate = json.loads((res / "n6c_gate.json").read_text())
    if not gate["passes"] and not smoke:
        raise SystemExit("N6c did not pass its seed-42 gate (ADDENDUM_3_N6C.md §3): no test")
    seeds = SMOKE_TEST_SEEDS if smoke else TEST_SEEDS
    pooled = {s: {m: [] for m in METRICS} for s in SCORERS}
    clusters, per_seed, picks_all, provs, arrays = [], {}, {}, {}, {}
    for seed in seeds:
        ctx, pa, picks, prov = score_seed(seed, smoke, res)
        per_seed[str(seed)] = block(pa, ctx.anchor_group)
        picks_all[str(seed)], provs[str(seed)] = picks, prov
        for s in SCORERS:
            for m in METRICS:
                arrays[f"seed{seed}__{s}__{m}"] = np.asarray(pa[s][m])
                pooled[s][m].append(np.asarray(pa[s][m], dtype=np.float64))
        arrays[f"seed{seed}__anchor_group"] = ctx.anchor_group
        clusters.append(ctx.anchor_group)
        rc.log(f"test seed {seed} done")
    cl = np.concatenate(clusters)
    P = {s: {m: np.concatenate(v) for m, v in d.items()} for s, d in pooled.items()}
    pb = block(P, cl)
    pb["n_episodes"], pb["n_clusters"] = int(len(cl)), int(len(np.unique(cl)))
    return {"seeds": list(seeds), "per_seed": per_seed, "picks": picks_all, "provenance": provs, "pooled": pb,
            "go": go_verdict(pb["vs"], ADDENDUM_COMPARATORS), "gate_passed": gate["passes"]}, arrays


def text(r, stage):
    def c(x):
        return f"{x['point']:6.2f} [{x['ci95'][0]:6.2f}, {x['ci95'][1]:6.2f}]"

    def lines_for(b):
        out = [f"  {s:7s} R@1 {c(b[s]['summary']['r1'])} gain {c(b[s]['summary']['gain'])} "
               f"either {b[s]['either']['point']:6.2f}" for s in SCORERS]
        out += [f"  config - {comp:7s} R@1 {c(b['vs'][comp]['r1'])} gain {c(b['vs'][comp]['gain'])}"
                for comp in ADDENDUM_COMPARATORS]
        return out

    smoke = " (SMOKE)" if r["smoke"] else ""
    if stage == "gate":
        out = [f"N6c gate, seed {r['seed']}{smoke}", *lines_for(r), f"  picks {r['picks']}"]
        for pair, v in r["per_pair_vs"].items():
            out.append(f"  {pair}: vs control R@1 {c(v['control']['r1'])} gain {c(v['control']['gain'])}; "
                       f"vs matched R@1 {c(v['matched']['r1'])}")
        out.append(f"GATE PASSED: {r['passes']}")
        return "\n".join(out)
    out = [f"N6c fresh-seed test, seeds {r['seeds']}{smoke}"]
    for seed in r["seeds"]:
        out += [f"{seed}:", *lines_for(r["per_seed"][str(seed)]), f"  picks {r['picks'][str(seed)]}"]
    out += ["pooled:", *lines_for(r["pooled"]), f"GO: {r['go']}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("gate", "test"), required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    res = HERE / "results" / ("smoke" if args.smoke else "")
    stem = f"n6c_{args.stage}"
    outs = [res / f"{stem}.json", res / f"{stem}.txt", res / f"per_anchor_{stem}.npz"]
    if not args.smoke and any(p.exists() for p in outs):
        raise SystemExit(f"{stem} results exist in {res}; refusing to overwrite")
    t0 = time.time()
    result, arrays = (gate_stage if args.stage == "gate" else test_stage)(args.smoke, res)
    result["smoke"] = args.smoke
    result["script_sha256"] = rg.sha_file(Path(__file__))
    result["addendum_sha256"] = rg.sha_file(HERE / "ADDENDUM_3_N6C.md")
    rg.assert_finite_tree(result)
    outs[0].write_text(json.dumps(result, indent=1))
    np.savez_compressed(outs[2], **arrays)
    t = text(result, args.stage)
    outs[1].write_text(t + "\n")
    print(t)
    rc.log(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run both smoke stages**

```bash
cd /project/CoSiR
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261108_new_method_quick_checks/run_n6c.py --stage gate --smoke 2>&1 | tee src/test/20261108_new_method_quick_checks/results/smoke_n6c_gate.log
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261108_new_method_quick_checks/run_n6c.py --stage test --smoke 2>&1 | tee src/test/20261108_new_method_quick_checks/results/smoke_n6c_test.log
```
Expected: the gate ends with `GATE PASSED: ...` and `runtime`; the test ends with `GO: ...` and `runtime`; `results/smoke/` holds `n6c_gate.*` and `n6c_test.*`. Smoke numbers mean nothing. An AssertionError is a defect to fix at its root.

- [ ] **Step 3: Commit**

```bash
git add src/test/20261108_new_method_quick_checks/run_n6c.py
git commit -m "feat(v2): N6c runner (gate vs nested and matched controls, gated fresh-seed test; smoke-tested)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VzPGkSgZ3UrCEi5JYqTDw1"
```
