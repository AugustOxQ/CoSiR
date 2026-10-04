# Fresh-seed test of N1-nested-A3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the fixed configuration N1-nested-A3 and its comparators (cosine, RCA, its own condition-free control) on the fresh episode seeds 45, 47 and 48, report each seed and the pooled result, and apply the pre-committed GO rule.

**Architecture:** One pure decision function `go_verdict` appended to `src/eval/aspect_quick_checks.py` (unit-tested), and one runner `src/test/20261108_new_method_quick_checks/test_seeds.py` that, per seed, loads E1's episodes through E3's `EvalContext`, encodes A3's codes, computes A′'s cross-fitted nested score with the centered term (`crossfit_nested(cos, T_u, T_N1, parity)`) and its control, takes cosine and RCA from E1's per-anchor arrays, then pools the three seeds with painting clusters.

**Tech Stack:** Python 3.10 (conda env `CoSiR`), numpy, torch (CPU through existing helpers), pytest.

**Spec:** `docs/superpowers/specs/2026-10-04-new-method-quick-checks-design.md` §5 row 1; the committed rule `src/test/20261108_new_method_quick_checks/DECISION_RULE.md` §6 (7e50f18) and the fixed configuration `src/test/20261108_new_method_quick_checks/TEST_CONFIG.md` (00f2b32). Read both before starting.

## Global Constraints

- Python: `/root/miniconda3/envs/CoSiR/bin/python`. Never install anything into the env.
- CPU only. Prefix every command with `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`.
- Tests: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_quick_checks.py -q` from `/project/CoSiR`.
- Implementers run only `--smoke`. The real test is launched by the controller once, after review.
- Git: branch `main`; stage files by explicit path only; commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01VzPGkSgZ3UrCEi5JYqTDw1`.
- Do not edit any existing file except appending to `src/eval/aspect_quick_checks.py` and `src/test/test_aspect_quick_checks.py` (both created on this branch). Never edit `DECISION_RULE.md` or `TEST_CONFIG.md`.
- GO (DECISION_RULE.md §6): on the three seeds pooled, every paired difference config − comparator, for R@1 and condition gain, against each of cosine, RCA and the own control, has a 95% lower bound above 0 (painting-clustered bootstrap, 5,000 resamples, seed 42; one cluster per painting across seeds). Each seed is reported on its own, descriptively.

## Review Focus

1. **Seed/array misalignment.** E1's per-anchor arrays for a seed must belong to the same episodes as the context; the runner asserts cosine equality plus `anchor_group` and `pair_index` equality per seed.
2. **Pooled clustering.** Pooling must concatenate per-anchor arrays and the global painting ids (`ctx.anchor_group`, from `artelingo_splits().groups`), so one painting is one cluster across seeds; pooling by seed-local ids would understate the intervals.
3. **Wrong checkpoint or rule.** A3's SHA-256 is asserted (real run); the score is `centered_term`, not `agreement_term`, inside `crossfit_nested`.
4. **Silent misses.** Every score dict is asserted finite before scoring.
5. **Re-running after reading.** The real run refuses to overwrite its results.

---

### Task 1: GO verdict function and the fresh-seed runner

**Files:**
- Modify: `src/eval/aspect_quick_checks.py` (append)
- Modify: `src/test/test_aspect_quick_checks.py` (append)
- Create: `src/test/20261108_new_method_quick_checks/test_seeds.py`

**Interfaces:**
- Consumes: from `src.eval.aspect_quick_checks`: `centered_term(inputs, ep, uniform=False) -> scores dict`; `src.eval.aspect_scorers.EvalInputs`, `agreement_term(inputs, ep, uniform)`; `src.eval.aspect_nested.crossfit_nested(cos, t_u, t_a, parity) -> (nested, control, picks)`; `src.eval.aspect_metrics.METRICS, per_anchor, compare, cluster_bootstrap, summarize`; `run_gonogo` as `rg` (`rg.EvalContext(seed, smoke)` with `.img`, `.txt`, `.encode(ckpt)`, `.pooled`, `.cos`, `.parity`, `.anchor_group`, `.pair_index`, `.shas`, `.n`, `.summary(pa)`; `rg.folders(smoke)["e1"]`, `rg.checkpoint_path(run, seed, smoke)`, `rg.sha_file`, `rg.assert_finite_tree`, `rg.POOLED_ORDER`).
- Produces: `GO_COMPARATORS = ("cosine", "rca", "control")`; `go_verdict(pooled: dict) -> {"go": bool, "failed": list[str]}` where `pooled[comparator][metric]` is a `compare()` result for metric in `("r1", "gain")`. Runner outputs `results/test_seeds.json`, `results/per_anchor_test_seeds.npz`, `results/test_seeds.txt` (smoke: `results/smoke/`).

- [ ] **Step 1: Write the failing test**

Append to `src/test/test_aspect_quick_checks.py`:

```python
# ---------------------------------------------------------------- fresh-seed GO rule (DECISION_RULE.md §6)

from src.eval.aspect_quick_checks import GO_COMPARATORS, go_verdict  # noqa: E402


def test_go_verdict_needs_all_six_lower_bounds_above_zero():
    ok = {c: {m: _r(0.5, 0.1, 0.9) for m in ("r1", "gain")} for c in GO_COMPARATORS}
    assert GO_COMPARATORS == ("cosine", "rca", "control")
    assert go_verdict(ok) == {"go": True, "failed": []}
    bad = {c: dict(v) for c, v in ok.items()}
    bad["rca"]["gain"] = _r(0.2, 0.0, 0.4)                       # a lower bound of exactly 0 fails
    bad["control"]["r1"] = _r(0.1, -0.05, 0.3)
    assert go_verdict(bad) == {"go": False, "failed": ["rca/gain", "control/r1"]}
    with pytest.raises(KeyError):
        go_verdict({"cosine": ok["cosine"]})                      # every comparator must be present
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_quick_checks.py -q`
Expected: collection error, `ImportError: cannot import name 'GO_COMPARATORS'`.

- [ ] **Step 3: Append the function**

Append to `src/eval/aspect_quick_checks.py`:

```python
GO_COMPARATORS = ("cosine", "rca", "control")


def go_verdict(pooled: dict) -> dict:
    """§6: GO iff, on the pooled test seeds, the paired difference config − comparator has a 95% lower bound above 0
    for R@1 and for condition gain against every comparator. ``pooled[comparator][metric]`` is a compare() result."""
    failed = [f"{c}/{m}" for c in GO_COMPARATORS for m in ("r1", "gain") if not pooled[c][m]["ci95"][0] > 0]
    return {"go": not failed, "failed": failed}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_quick_checks.py -q`
Expected: 24 passed.

- [ ] **Step 5: Write the runner**

Create `src/test/20261108_new_method_quick_checks/test_seeds.py`:

```python
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
```

- [ ] **Step 6: Run the smoke run**

Run from `/project/CoSiR`:
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261108_new_method_quick_checks/test_seeds.py --smoke 2>&1 | tee src/test/20261108_new_method_quick_checks/results/smoke_test_seeds.log`
Expected: ends with `GO: ...` and `runtime ...s`; `results/smoke/` holds `test_seeds.json`, `per_anchor_test_seeds.npz`, `test_seeds.txt`. Smoke numbers mean nothing. An `AssertionError` is a defect to fix at its root, never to silence.

- [ ] **Step 7: Check the smoke outputs**

```bash
cd /project/CoSiR && /root/miniconda3/envs/CoSiR/bin/python -c "
import json, numpy as np
r = json.load(open('src/test/20261108_new_method_quick_checks/results/smoke/test_seeds.json'))
z = np.load('src/test/20261108_new_method_quick_checks/results/smoke/per_anchor_test_seeds.npz')
assert r['seeds'] == [42, 43] and set(r['per_seed']) == {'42', '43'}
n = sum(len(z[f'seed{s}__anchor_group']) for s in (42, 43))
assert r['pooled']['n_episodes'] == n
assert set(r['pooled']['vs']) == {'cosine', 'rca', 'control'}
assert all(np.isfinite(z[k]).all() for k in z.files)
print('smoke outputs OK', r['go'])
"
```
Expected: `smoke outputs OK {...}`.

- [ ] **Step 8: Commit**

```bash
git add src/eval/aspect_quick_checks.py src/test/test_aspect_quick_checks.py src/test/20261108_new_method_quick_checks/test_seeds.py
git commit -m "feat(v2): fresh-seed test runner for N1-nested-A3 and the GO verdict (smoke-tested)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VzPGkSgZ3UrCEi5JYqTDw1"
```
