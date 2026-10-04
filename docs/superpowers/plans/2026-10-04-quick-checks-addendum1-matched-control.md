# Addendum 1: N1's matched condition-free control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ADDENDUM_1.md's matched condition-free control for N1, apply it to the seed-42 checks (R2: re-apply the decision table) and to the stored fresh-seed test (R3: a fourth GO comparator).

**Architecture:** One unit-tested function `crossfit_condition_free` (and a `comparators` parameter on `go_verdict`) in `src/eval/aspect_quick_checks.py`; one runner `src/test/20261108_new_method_quick_checks/matched_controls.py` that recomputes N1-nested (asserting it equals the stored arrays), computes the matched control, and writes `results/addendum1.*`.

**Tech Stack:** Python 3.10 (conda env `CoSiR`), numpy, pytest.

**Spec:** `src/test/20261108_new_method_quick_checks/ADDENDUM_1.md` (812cab2), which amends `DECISION_RULE.md` (7e50f18) and `TEST_CONFIG.md` (00f2b32). Read all three.

## Global Constraints

- Python: `/root/miniconda3/envs/CoSiR/bin/python`; never install anything.
- CPU only: prefix every command with `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`.
- Tests: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_quick_checks.py -q` from `/project/CoSiR`.
- Implementers run only `matched_controls.py --smoke`. Never run `run_checks.py` or `test_seeds.py` without `--smoke`; never read or open `results/test_seeds.*`, `results/per_anchor_test_seeds.npz` or `results/test_seeds.log` (the controller runs the real script).
- Git: `main`; stage by explicit path; commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01VzPGkSgZ3UrCEi5JYqTDw1`.
- Edit only the files listed in the task. Never edit DECISION_RULE.md, TEST_CONFIG.md or ADDENDUM_1.md.
- Matched control (ADDENDUM_1 R1): the 56-cell family z(cos) + λ_u·z(T_u) + λ_a·z(T_N1u), T_N1u = `centered_term(..., uniform=True)`, λ grids of `src.eval.aspect_nested` (`nested_cells()`), each episode-index parity half picks the cell with the highest R@1 (ties to the first cell in row-major order, λ_u outer), each half's pick scores the other half.

## Review Focus

1. **A condition-dependent term passed as a control term** would leak the condition into the control; `crossfit_condition_free` raises if `t_u` or `t_c` differs between conditions (unit-tested).
2. **Recomputed N1-nested drifting from the stored arrays** would compare a different configuration; the runner asserts bit-equality with `run_checks.py`'s and `test_seeds.py`'s stored per-anchor arrays.
3. **Pooled comparisons inconsistent with test_seeds.py** for the three original comparators: the runner asserts its recomputed pooled points equal the stored ones.
4. **Out-of-sample cross-fit:** half 0 must be scored with half 1's pick (unit-tested).
5. **Overwriting results:** the real run refuses to overwrite `results/addendum1.*`.

---

### Task 1: Matched-control cross-fit and the addendum runner

**Files:**
- Modify: `src/eval/aspect_quick_checks.py` (top imports; `go_verdict` signature; append)
- Modify: `src/test/test_aspect_quick_checks.py` (append)
- Create: `src/test/20261108_new_method_quick_checks/matched_controls.py`

**Interfaces:**
- Consumes: `src.eval.aspect_nested.nested_cells() -> list[(λ_u, λ_a)]`, `nested_scores(cos, t_u, t_a, lam_u, lam_a) -> scores dict`, `crossfit_nested`; `src.eval.aspect_metrics.per_anchor, compare, summarize, METRICS`; `run_checks.py` as `rc` (`rc.rg`, `rc.model_inputs(ctx, name, scorer_train, smoke) -> (EvalInputs, scale, prov)`, `rc.point_ci(values, clusters)`, `rc.A3_SHA`).
- Produces: `ADDENDUM_COMPARATORS = ("cosine", "rca", "control", "matched")`; `go_verdict(pooled, comparators=GO_COMPARATORS)`; `crossfit_condition_free(cos, t_u, t_c, parity) -> (scores dict, {0: [λ_u, λ_a], 1: [λ_u, λ_a]})`.

- [ ] **Step 1: Write the failing tests**

Append to `src/test/test_aspect_quick_checks.py`:

```python
# ---------------------------------------------------------------- ADDENDUM_1.md: matched condition-free control

from src.eval.aspect_nested import nested_cells, nested_scores  # noqa: E402
from src.eval.aspect_quick_checks import ADDENDUM_COMPARATORS, crossfit_condition_free  # noqa: E402


def _cf_world(n=400, seed=3):
    """Condition-free scores: cosine and t_u random; t_c pushes column 0 up for most episodes."""
    rng = np.random.default_rng(seed)
    cos, t_u = rng.normal(size=(n, 13)), rng.normal(size=(n, 13))
    t_c = rng.normal(size=(n, 13))
    t_c[:, 0] += 3.0 * (rng.random(n) < 0.7)
    return _sc(cos), _sc(t_u), _sc(t_c), np.arange(n) % 2


def _r1_on(scores, rows):
    return float(per_anchor({c: {d: scores[c][d][rows] for d in DIRECTIONS} for c in CONDITIONS})["r1"].mean())


def test_crossfit_condition_free_picks_the_first_best_cell_and_applies_it_out_of_sample():
    cos, t_u, t_c, parity = _cf_world()
    out, picks = crossfit_condition_free(cos, t_u, t_c, parity)
    for half in (0, 1):
        tune = parity == half
        r1 = [_r1_on(nested_scores(cos, t_u, t_c, *cell), tune) for cell in nested_cells()]
        best = nested_cells()[int(np.argmax(r1))]                    # argmax keeps the first maximum
        assert picks[half] == [best[0], best[1]]
        applied = nested_scores(cos, t_u, t_c, *best)
        for c in CONDITIONS:
            for d in DIRECTIONS:
                np.testing.assert_array_equal(out[c][d][parity != half], applied[c][d][parity != half])
    assert picks[0][1] > 0                                           # the column-0 term is used
    assert (per_anchor(out)["gain"] == 0).all()                      # condition-free by construction


def test_crossfit_condition_free_rejects_a_conditioned_term():
    cos, t_u, t_c, parity = _cf_world(n=40)
    t_c["b"]["i2t"] = t_c["b"]["i2t"] + 1e-3
    with pytest.raises(ValueError):
        crossfit_condition_free(cos, t_u, t_c, parity)
    with pytest.raises(ValueError):
        crossfit_condition_free(cos, t_c, t_u, parity)
    with pytest.raises(ValueError):
        crossfit_condition_free(cos, t_u, _cf_world(n=40)[2], np.zeros(40))      # one empty half


def test_go_verdict_with_the_matched_comparator():
    assert ADDENDUM_COMPARATORS == ("cosine", "rca", "control", "matched")
    ok = {c: {m: _r(0.5, 0.1, 0.9) for m in ("r1", "gain")} for c in ADDENDUM_COMPARATORS}
    assert go_verdict(ok, ADDENDUM_COMPARATORS) == {"go": True, "failed": []}
    ok["matched"]["r1"] = _r(-0.03, -0.07, 0.0)
    assert go_verdict(ok, ADDENDUM_COMPARATORS) == {"go": False, "failed": ["matched/r1"]}
    assert go_verdict(ok) == {"go": True, "failed": []}                          # §6 comparators only
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_quick_checks.py -q`
Expected: collection error, `ImportError: cannot import name 'ADDENDUM_COMPARATORS'`.

- [ ] **Step 3: Implement**

In `src/eval/aspect_quick_checks.py`, replace the import line

```python
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS
```

with

```python
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.eval.aspect_nested import nested_cells, nested_scores
```

replace the existing `go_verdict` definition (keep `GO_COMPARATORS` above it unchanged) with

```python
def go_verdict(pooled: dict, comparators=GO_COMPARATORS) -> dict:
    """§6 (and ADDENDUM_1 R3 with comparators=ADDENDUM_COMPARATORS): GO iff, on the pooled test seeds, the paired
    difference config − comparator has a 95% lower bound above 0 for R@1 and for condition gain against every
    comparator. ``pooled[comparator][metric]`` is a compare() result."""
    failed = [f"{c}/{m}" for c in comparators for m in ("r1", "gain") if not pooled[c][m]["ci95"][0] > 0]
    return {"go": not failed, "failed": failed}
```

and append at the end of the file

```python
# ---------------------------------------------------------------- ADDENDUM_1.md: matched condition-free control

ADDENDUM_COMPARATORS = (*GO_COMPARATORS, "matched")


def _require_condition_free(scores: dict, what: str) -> None:
    for d in DIRECTIONS:
        if not np.array_equal(np.asarray(scores["a"][d]), np.asarray(scores["b"][d]), equal_nan=True):
            raise ValueError(f"{what} must be identical under both conditions (condition-free)")


def crossfit_condition_free(cos: dict, t_u: dict, t_c: dict, parity) -> tuple:
    """ADDENDUM_1 R1: the matched condition-free control. Over the 56 nested cells z(cos) + λ_u·z(t_u) + λ_a·z(t_c),
    with t_u and t_c both condition-free, each episode-index parity half picks the cell with the highest R@1 (ties to
    the first cell in row-major order, λ_u outer); each half's pick scores the other half."""
    _require_condition_free(t_u, "t_u")
    _require_condition_free(t_c, "t_c")
    n = len(cos["a"]["i2t"])
    parity = np.asarray(parity)
    if parity.shape != (n,) or not np.isin(parity, (0, 1)).all() or not ((parity == 0).any() and (parity == 1).any()):
        raise ValueError(f"parity must be a length-{n} array of 0/1 with both halves non-empty")
    cache = {}

    def scored(cell):
        if cell not in cache:
            cache[cell] = nested_scores(cos, t_u, t_c, *cell)
        return cache[cell]

    def r1(cell, rows):
        s = scored(cell)
        return float(per_anchor({c: {d: s[c][d][rows] for d in DIRECTIONS} for c in CONDITIONS})["r1"].mean())

    out = {c: {d: np.empty(np.asarray(cos[c][d]).shape, np.float32) for d in DIRECTIONS} for c in CONDITIONS}
    picks = {}
    for half in (0, 1):
        tune, apply = parity == half, parity != half
        cell = max(nested_cells(), key=lambda c: r1(c, tune))        # max keeps the first of equal maxima
        picks[half] = [float(cell[0]), float(cell[1])]
        for c in CONDITIONS:
            for d in DIRECTIONS:
                out[c][d][apply] = scored(cell)[c][d][apply]
    return out, picks
```

- [ ] **Step 4: Run the tests to verify they pass**

Run the test command. Expected: 27 passed.

- [ ] **Step 5: Write the runner**

Create `src/test/20261108_new_method_quick_checks/matched_controls.py`:

```python
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
```

- [ ] **Step 6: Run the smoke run**

`cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261108_new_method_quick_checks/matched_controls.py --smoke 2>&1 | tee src/test/20261108_new_method_quick_checks/results/smoke_addendum1.log`
Expected: ends with the two GO lines and `runtime ...s`; `results/smoke/` holds `addendum1.json`, `addendum1.txt`, `per_anchor_addendum1.npz`. Smoke numbers mean nothing. An AssertionError is a defect to fix at its root, never to silence; if `test_seeds.json` or the stored arrays disagree, report it.

- [ ] **Step 7: Commit**

```bash
git add src/eval/aspect_quick_checks.py src/test/test_aspect_quick_checks.py src/test/20261108_new_method_quick_checks/matched_controls.py
git commit -m "feat(v2): ADDENDUM_1 matched condition-free control for N1 (cross-fit, GO with four comparators, runner; smoke-tested)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VzPGkSgZ3UrCEi5JYqTDw1"
```
