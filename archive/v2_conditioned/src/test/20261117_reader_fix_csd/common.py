"""Shared code of the reader-fix round (binding spec: DECISION_RULE.md in this folder, SHA-256 asserted).

Import-light on purpose: numpy, torch and the repo's eval modules only. Everything that needs the real data or the older
exploratory scripts (run_step1, run_sweep, run_told_oracle) is loaded inside `load_bundle`. The pure functions
(`pair_agreements`, `hard_term`, `bar_info`, `evaluate_fused`, ...) take plain arrays and are unit-tested with synthetic
data in test_reader_fix.py. CPU only.

API used by run_ra.py, run_rc.py and the R-b scripts: RULE_SHA, assert_rule, CONFIGS, TOLD, load_bundle,
pair_agreements, support_argmax_match, grouping_stack, hard_term, expected_term, evaluate, evaluate_fused,
save_candidate, paired_diff.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, cluster_bootstrap, compare, per_anchor  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import (_require_condition_free, _stacked_dots,  # noqa: E402
                                          crossfit_condition_free)

RULE = HERE / "DECISION_RULE.md"
RULE_SHA = "613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c"
RES = HERE / "results"

GROUPINGS = ("affect", "image", "caption", "csd", "rand")
CONFIGS = {"A0": ("affect", "image", "caption"),
           "A1": ("affect", "image", "caption", "csd"),
           "AR": ("affect", "image", "caption", "rand")}
# D13: told mapping (diagnostic only), aspect -> grouping
TOLD = {"A0": {"emotion": "affect", "style": "image", "genre": "image"},
        "A1": {"emotion": "affect", "style": "csd", "genre": "image"},
        "AR": {"emotion": "affect", "style": "rand", "genre": "image"}}
PAIRS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))
POOLED_ORDER = [f"{a}__{b}" for a, b in PAIRS]
STEP1_NAME = {"affect": "affect", "image": "image", "caption": "caption", "csd": "style_csd", "rand": "style_rand"}
BAR_TARGET = 0.5          # D12, percentage points
SMOKE_PER_PAIR = 200      # smoke runs use the first 200 episodes of each aspect pair (600 in all; both parities)

_T = ROOT / "src/test"
INPUT_FILES = {
    "decision_rule": RULE,
    "episodes_seed42": _T / "20261030_aspect_baselines/results/episodes_seed42.npz",
    "affect_L_partition(per_anchor_told_oracle)": _T / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz",
    "e2_partitions": _T / "20261031_pseudo_partitions/results/partitions.npz",
    "step1_group_style": _T / "20261116_grouping_step1_style/results/step1_group_style.npz",
    "n6_posteriors": _T / "20261108_new_method_quick_checks/results/n6_posteriors.npz",
    "step1_heads_style": _T / "20261116_grouping_step1_style/results/step1_heads_style.npz",
    "step1_eval_style": _T / "20261116_grouping_step1_style/results/step1_eval_style.npz",
    "method_A_checkpoint": _T / "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt",
}
INPUT_SHA = {
    "decision_rule": RULE_SHA,
    "episodes_seed42": "12af979432ff1a20c88b614ab9e672f203eeeed9c2465bff305e72d01e6c0986",
    "affect_L_partition(per_anchor_told_oracle)": "27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366",
    "e2_partitions": "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa",
    "step1_group_style": "b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2",
    "n6_posteriors": "2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0",
    "step1_heads_style": "898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b",
    "step1_eval_style": "8d10a0fbbd34212c73849239faed9d0a07372e68732dc452bf2fd57f7409c68c",
    "method_A_checkpoint": "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2",
}


# ---------------------------------------------------------------- small utilities

def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def now_ams() -> str:
    """Amsterdam local time, plain (no offset), as the project's timestamp rule asks."""
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M")


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable ({e})"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def assert_rule():
    """Every script calls this first: the rule's SHA-256 must be the committed one."""
    got = sha_file(RULE)
    if got != RULE_SHA:
        raise SystemExit(f"DECISION_RULE.md differs from the committed version: {got} != {RULE_SHA}")


_INPUT_CHECKED = {}


def verify_inputs() -> dict:
    """SHA-256 of every input file the rule names (D1, D2, section 2); a mismatch stops the run."""
    if _INPUT_CHECKED:
        return dict(_INPUT_CHECKED)
    for k, p in INPUT_FILES.items():
        got = sha_file(p)
        if got != INPUT_SHA[k]:
            raise SystemExit(f"input {k} ({p}): SHA-256 {got} differs from the rule's {INPUT_SHA[k]}")
        _INPUT_CHECKED[k] = got
    return dict(_INPUT_CHECKED)


def point_ci(values, clusters) -> dict:
    """Mean in percentage points with the painting-clustered 95% interval (5,000 resamples, seed 42)."""
    r = cluster_bootstrap(values, clusters)
    return {"point": 100 * r["point"], "ci95": [100 * c for c in r["ci95"]]}


def either(pa):
    return np.asarray(pa["r1"], dtype=np.float64) + np.asarray(pa["other"], dtype=np.float64)


def diff3(pa, pb, cl) -> dict:
    """Paired per-anchor difference in R@1, condition gain and either rate."""
    return {"r1": compare(pa, pb, cl, "r1"), "gain": compare(pa, pb, cl, "gain"),
            "either": point_ci(either(pa) - either(pb), cl)}


def sub(pa, mask):
    return {m: np.asarray(pa[m])[mask] for m in METRICS}


def cf_version(t):
    """diagnose_counterparts.cf_version (identical; load_bundle asserts it against the original)."""
    return {cnd: {d: (0.5 * (t["a"][d].astype(np.float64) + t["b"][d].astype(np.float64))).astype(np.float32)
                  for d in DIRECTIONS} for cnd in CONDITIONS}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def assert_finite_tree(obj, where=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_finite_tree(v, f"{where}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_finite_tree(v, f"{where}[{i}]")
    elif isinstance(obj, float) and not np.isfinite(obj):
        raise AssertionError(f"non-finite number at {where}")


# ---------------------------------------------------------------- agreements, stacks, terms (D3 to D5)

def pair_agreements(post, si, st, ci, ct, parts):
    """(sup, con), each (E, 4, H): a_h = p_h(image) . p_h(caption) per support / contrast pair (D3). si, st, ci, ct are
    the global-row index arrays (E, 4) that `episodes.condition(cond)` returns."""
    def agree(ii, tt):
        return np.stack([np.einsum("nsc,nsc->ns", post[h]["img"][ii], post[h]["txt"][tt]) for h in parts], axis=2)
    return agree(si, st), agree(ci, ct)


def support_argmax_match(post, si, st, parts):
    """(E, 4, H) bool: the support pair's image and caption posteriors have the same arg-max group (R-b feature)."""
    return np.stack([post[h]["img"][si].argmax(axis=-1) == post[h]["txt"][st].argmax(axis=-1) for h in parts], axis=2)


def grouping_stack(post, ep, parts):
    """{dir: (E, H, K)} of s_h = p_h(query) . p_h(candidate) (D4), as aspect_quick_checks._stacked_dots."""
    return _stacked_dots(post, ep, tuple(parts))


def hard_term(stack, picks):
    """D5 hard reader: T^c = s_{pi^c}. picks = {"a": (E,), "b": (E,)} ints. -> {cond: {dir: (E, K) float32}}."""
    out = {}
    for c in CONDITIONS:
        idx = np.asarray(picks[c], dtype=np.int64)
        rows = np.arange(len(idx))
        out[c] = {d: np.asarray(stack[d][rows, idx], dtype=np.float32) for d in DIRECTIONS}
    return out


def expected_term(stack, probs):
    """R-b expected: T^c = sum_h P^c(h) s_h. probs = {"a": (E, H), "b": (E, H)}."""
    return {c: {d: np.einsum("nh,nhk->nk", np.asarray(probs[c], dtype=np.float64),
                             np.asarray(stack[d], dtype=np.float64)).astype(np.float32) for d in DIRECTIONS}
            for c in CONDITIONS}


def top_two_margin(scores) -> np.ndarray:
    """(E,): largest minus second-largest of an (E, H) score array."""
    s = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    return s[:, -1] - s[:, -2]


# ---------------------------------------------------------------- comparators, bar margin, pick statistics

def bar_comparator(pBp, pc, pB):
    """D10: of B', the matched counterpart and B, the one with the largest mean R@1; ties to the earliest in the order
    B', counterpart, B (full precision)."""
    cands = (("B_prime", pBp), ("counterpart", pc), ("B", pB))
    means = [float(np.mean(np.asarray(p["r1"], dtype=np.float64))) for _, p in cands]
    best = 0
    for i in range(1, 3):
        if means[i] > means[best]:
            best = i
    return cands[best][0], cands[best][1], {n: m for (n, _), m in zip(cands, means)}


def bar_info(pn, pc, pBp, pB, cl, pair_index):
    """D10/D11: bar margin (fused minus bar comparator, paired per anchor), per aspect pair with the same comparator."""
    name, comp, means = bar_comparator(pBp, pc, pB)
    v = np.asarray(pn["r1"], dtype=np.float64) - np.asarray(comp["r1"], dtype=np.float64)
    g = np.asarray(pn["gain"], dtype=np.float64) - np.asarray(comp["gain"], dtype=np.float64)
    info = {"comparator": name, "comparator_mean_r1": {k: 100 * m for k, m in means.items()},
            "fused_r1": 100 * float(np.mean(pn["r1"])), "r1": point_ci(v, cl), "gain_vs_comparator": point_ci(g, cl),
            "per_pair_r1": {p: point_ci(v[pair_index == i], cl[pair_index == i]) for i, p in enumerate(POOLED_ORDER)},
            "per_pair_gain": {p: point_ci(g[pair_index == i], cl[pair_index == i])
                              for i, p in enumerate(POOLED_ORDER)}}
    return v, info


def clears_bar(bar_r1, gain_stat) -> dict:
    """D12 at full precision."""
    c1 = bool(bar_r1["point"] >= BAR_TARGET)
    c2 = bool(bar_r1["ci95"][0] > 0)
    c3 = bool(gain_stat["ci95"][0] > 0)
    return {"clause1_bar_point_at_least_0.5": c1, "clause2_bar_lower_above_0": c2,
            "clause3_gain_statistic_lower_above_0": c3, "clears_bar": bool(c1 and c2 and c3)}


def told_index(parts, told, pair_index):
    """{cond: (E,)} index (in configuration order) of the told grouping of the aspect each condition is about."""
    parts = tuple(parts)
    out = {}
    for j, c in enumerate(CONDITIONS):
        out[c] = np.array([parts.index(told[PAIRS[i][j]]) for i in pair_index], dtype=np.int64)
    return out


def pick_statistics(picks, parts, told, pair_index, cl):
    """Pick accuracy under the told mapping (D13) and the share of picks per grouping, overall and per pair/condition."""
    parts = tuple(parts)
    ti = told_index(parts, told, pair_index)
    correct = {c: np.asarray(picks[c]) == ti[c] for c in CONDITIONS}
    both = correct["a"] & correct["b"]
    per_ep = 0.5 * (correct["a"].astype(np.float64) + correct["b"].astype(np.float64))
    acc = {"correct_share": point_ci(per_ep, cl), "both_correct_share": 100 * float(both.mean()),
           "per_pair_condition": {p: {c: 100 * float(correct[c][pair_index == i].mean()) for c in CONDITIONS}
                                  for i, p in enumerate(POOLED_ORDER)},
           "chance": 100.0 / len(parts)}
    return acc, pick_shares(picks, parts, pair_index)


def pick_shares(picks, parts, pair_index):
    """Share (%) of (episode, condition) picks going to each grouping: overall, per condition, per pair and condition."""
    parts = tuple(parts)
    allp = np.concatenate([np.asarray(picks[c]) for c in CONDITIONS])
    return {"overall": {h: 100 * float(np.mean(allp == j)) for j, h in enumerate(parts)},
            "per_condition": {c: {h: 100 * float(np.mean(np.asarray(picks[c]) == j)) for j, h in enumerate(parts)}
                              for c in CONDITIONS},
            "per_pair_condition": {p: {c: {h: 100 * float(np.mean(np.asarray(picks[c])[pair_index == i] == j))
                                           for j, h in enumerate(parts)} for c in CONDITIONS}
                                   for i, p in enumerate(POOLED_ORDER)}}


# ---------------------------------------------------------------- evaluation of a candidate

def per_pair_summary(pn, pc, pB, pBp, comp_name, cl, pair_index):
    out = {}
    comp = {"B_prime": pBp, "counterpart": pc, "B": pB}[comp_name]
    for i, p in enumerate(POOLED_ORDER):
        m = pair_index == i
        a, b, base, bp, cp = sub(pn, m), sub(pc, m), sub(pB, m), sub(pBp, m), sub(comp, m)
        out[p] = {"margin": {"r1": compare(a, b, cl[m], "r1"), "gain": compare(a, b, cl[m], "gain"),
                             "either": point_ci(either(a) - either(b), cl[m])},
                  "fused_vs_B_r1": compare(a, base, cl[m], "r1"), "fused_vs_Bprime_r1": compare(a, bp, cl[m], "r1"),
                  "counterpart_vs_B_r1": compare(b, base, cl[m], "r1"),
                  "bar_margin_r1": compare(a, cp, cl[m], "r1")}
    return out


def evaluate_fused(bundle, config, pn, pc, picks, name, crossfit=None):
    """Summary of a candidate from its already cross-fitted per-anchor arrays: pn (fused reader) and pc (fused matched
    counterpart). picks = {"a": (E,), "b": (E,)} (the reader's grouping picks, for pick accuracy; None to skip)."""
    ctx = bundle.ctx
    cl, pi = ctx.anchor_group, ctx.pair_index
    pB, pBp = bundle.pB, bundle.pBp[config]
    margin = diff3(pn, pc, cl)
    bar_v, bar = bar_info(pn, pc, pBp, pB, cl, pi)
    gain_stat = margin["gain"]                                    # fusedT_vs_fusedTcf.gain (D11)
    summary = {"name": name, "config": config, "groupings": list(CONFIGS[config]), "n_episodes": int(len(cl)),
               "n_clusters": int(len(np.unique(cl))),
               "r1_means": {"fused": 100 * float(np.mean(pn["r1"])), "counterpart": 100 * float(np.mean(pc["r1"])),
                            "B": 100 * float(np.mean(pB["r1"])), "B_prime": 100 * float(np.mean(pBp["r1"]))},
               "margin": margin, "fused_vs_B": diff3(pn, pB, cl), "fused_vs_Bprime": diff3(pn, pBp, cl),
               "counterpart_vs_B": diff3(pc, pB, cl), "bar": bar, "gain_statistic": gain_stat,
               "clears_bar": clears_bar(bar["r1"], gain_stat),
               "per_pair": per_pair_summary(pn, pc, pB, pBp, bar["comparator"], cl, pi),
               "crossfit": crossfit}
    arrays = {"fused": pn, "cf": pc, "bar_v": bar_v}
    if picks is not None:
        acc, share = pick_statistics(picks, CONFIGS[config], TOLD[config], pi, cl)
        summary["pick_accuracy"], summary["pick_share"] = acc, share
        summary["picks"] = {c: np.bincount(np.asarray(picks[c]), minlength=len(CONFIGS[config])).tolist()
                            for c in CONDITIONS}
    ref = getattr(bundle, "argmax", {}).get(config)
    if ref is not None:
        rn, rcf = ref["fused"], ref["cf"]
        summary["vs_step1_argmax"] = {
            "fused_r1": compare(pn, rn, cl, "r1"),
            "margin_r1": point_ci((np.asarray(pn["r1"]) - np.asarray(pc["r1"]))
                                  - (np.asarray(rn["r1"]) - np.asarray(rcf["r1"])), cl),
            "bar_margin_r1": point_ci(bar_v - ref["bar_v"], cl),
            "argmax_reader": {"fused_r1": 100 * float(np.mean(rn["r1"])), "bar_comparator": ref["bar"]["comparator"],
                              "bar_r1": ref["bar"]["r1"], "margin_r1": ref["margin_r1"]}}
    arrays["vec"] = {"fused_r1": np.asarray(pn["r1"], dtype=np.float64), "fused_gain": np.asarray(pn["gain"], np.float64),
                     "fused_either": either(pn), "margin_r1": np.asarray(pn["r1"], np.float64) - np.asarray(pc["r1"], np.float64),
                     "margin_gain": np.asarray(pn["gain"], np.float64) - np.asarray(pc["gain"], np.float64),
                     "bar_r1": bar_v}
    return summary, arrays


def evaluate(bundle, config, T, picks, name):
    """Fused reader crossfit_nested(B, B, T, parity); counterpart crossfit_condition_free(B, B, cf_version(T), parity).
    T = {cond: {dir: (E, K)}}; picks = {"a","b"} (E,) ints under the configuration's grouping order."""
    ctx = bundle.ctx
    _assert_finite(T, name)
    nested, ctrl, t_picks = crossfit_nested(bundle.B, bundle.B, T, ctx.parity)
    cf, cf_picks = crossfit_condition_free(bundle.B, bundle.B, cf_version(T), ctx.parity)
    pn, pc = per_anchor(nested), per_anchor(cf)
    ctrl_ok = all(np.array_equal(np.asarray(per_anchor(ctrl)[m]), np.asarray(bundle.pB[m])) for m in METRICS)
    return evaluate_fused(bundle, config, pn, pc, picks, name,
                          crossfit={"T_picks": t_picks, "cf_picks": cf_picks, "control_ranks_as_B": bool(ctrl_ok)})


def _assert_finite(scores, what):
    for c in scores:
        for d in scores[c]:
            if not np.isfinite(np.asarray(scores[c][d])).all():
                raise AssertionError(f"{what}: non-finite scores in {c}/{d}")


def paired_diff(arrays_x, arrays_y, metric, cl) -> dict:
    """Paired per-anchor difference x - y with its interval (pp). metric: fused_r1, fused_gain, fused_either,
    margin_r1, margin_gain or bar_r1 (the arrays are those `evaluate` / `evaluate_fused` return)."""
    return point_ci(arrays_x["vec"][metric] - arrays_y["vec"][metric], cl)


# ---------------------------------------------------------------- saving

def _paths(name, smoke):
    d = RES / "smoke" if smoke else RES
    return d, {k: d / f"cand_{name}.{k}" for k in ("json", "txt", "npz")}


def _script_path():
    return Path(getattr(sys.modules.get("__main__"), "__file__", __file__)).resolve()


def provenance(smoke, extra_inputs=None):
    sp = _script_path()
    inputs = dict(_INPUT_CHECKED) if _INPUT_CHECKED else verify_inputs()
    if extra_inputs:
        inputs.update(extra_inputs)
    return {"rule_sha256": RULE_SHA, "git_head": git_head(), "script": sp.name, "script_sha256": sha_file(sp),
            "common_sha256": sha_file(Path(__file__)), "time_amsterdam": now_ams(), "smoke": bool(smoke),
            "inputs_sha256": inputs, "argv": sys.argv,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}


def _fmt(x):
    return f"{x['point']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def summary_text(s) -> str:
    bar, cb = s["bar"], s["clears_bar"]
    L = [f"{s['name']} ({s['config']}; groupings {s['groupings']}){'  [SMOKE: not a result]' if s.get('smoke') else ''}",
         f"  R@1 means (pp): fused {s['r1_means']['fused']:.3f}, counterpart {s['r1_means']['counterpart']:.3f}, "
         f"B {s['r1_means']['B']:.3f}, B' {s['r1_means']['B_prime']:.3f}",
         f"  margin (fused - counterpart): R@1 {_fmt(s['margin']['r1'])}, gain {_fmt(s['margin']['gain'])}, "
         f"either {_fmt(s['margin']['either'])}",
         f"  bar margin R@1 {_fmt(bar['r1'])} vs {bar['comparator']} "
         f"(means {', '.join(f'{k} {v:.3f}' for k, v in bar['comparator_mean_r1'].items())})",
         f"  gain statistic {_fmt(s['gain_statistic'])}",
         f"  clears_bar {cb['clears_bar']} (>=0.5 {cb['clause1_bar_point_at_least_0.5']}, bar lower>0 "
         f"{cb['clause2_bar_lower_above_0']}, gain lower>0 {cb['clause3_gain_statistic_lower_above_0']})"]
    if "pick_accuracy" in s:
        pa = s["pick_accuracy"]
        L.append(f"  pick accuracy {_fmt(pa['correct_share'])} (chance {pa['chance']:.1f}); per pair a/b: " + "; ".join(
            f"{p} {v['a']:.1f}/{v['b']:.1f}" for p, v in pa["per_pair_condition"].items()))
    if "vs_step1_argmax" in s:
        v = s["vs_step1_argmax"]
        L.append(f"  vs step-1 arg-max: margin R@1 {_fmt(v['margin_r1'])}, bar margin {_fmt(v['bar_margin_r1'])}")
    return "\n".join(L)


def save_candidate(name, summary, arrays, T, picks, margins, extra=None, smoke=False):
    """results/cand_<name>.{json,txt,npz}. Refuses to overwrite a non-smoke file; smoke goes to results/smoke/.
    extra: {key: ndarray} goes to the npz (prefix extra__), anything else to the JSON under "extra"."""
    d, p = _paths(name, smoke)
    if not smoke and any(q.exists() for q in p.values()):
        raise SystemExit(f"results for candidate {name} exist in {d}; refusing to overwrite")
    d.mkdir(parents=True, exist_ok=True)
    npz = {}
    for part in ("fused", "cf"):
        for m in METRICS:
            npz[f"{part}__{m}"] = np.asarray(arrays[part][m])
    npz["bar_v"] = np.asarray(arrays["bar_v"])
    for c in CONDITIONS:
        for dr in DIRECTIONS:
            npz[f"T__{c}__{dr}"] = np.asarray(T[c][dr], dtype=np.float32)
        npz[f"pick__{c}"] = np.asarray(picks[c]).astype(np.int8)
        npz[f"margin__{c}"] = np.asarray(margins[c], dtype=np.float64)
    json_extra = {}
    for k, v in (extra or {}).items():
        if isinstance(v, np.ndarray):
            npz[f"extra__{k}"] = v
        else:
            json_extra[k] = v
    np.savez_compressed(p["npz"], **npz)
    rec = {**jsonable(summary), "smoke": bool(smoke), "extra": jsonable(json_extra),
           "provenance": {**provenance(smoke), "npz_sha256": sha_file(p["npz"])}}
    assert_finite_tree(rec)
    p["json"].write_text(json.dumps(rec, indent=1))
    txt = summary_text(rec)
    p["txt"].write_text(txt + "\n")
    print(txt, flush=True)
    return p


def load_candidate(name, smoke=False):
    """(json record, npz) of a saved candidate (SHA-256 of the npz checked against its JSON)."""
    _, p = _paths(name, smoke)
    if not p["json"].exists():
        raise SystemExit(f"{p['json']} is missing")
    rec = json.loads(p["json"].read_text())
    if rec["provenance"]["rule_sha256"] != RULE_SHA:
        raise SystemExit(f"{p['json']} was written under another rule")
    if sha_file(p["npz"]) != rec["provenance"]["npz_sha256"]:
        raise SystemExit(f"{p['npz']} differs from its JSON (SHA-256)")
    return rec, np.load(p["npz"])


def write_json_once(path, rec, smoke):
    """Write a standalone results JSON (non-smoke: never overwritten)."""
    path = Path(path)
    if not smoke and path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {**jsonable(rec), "provenance": provenance(smoke)}
    assert_finite_tree(rec)
    path.write_text(json.dumps(rec, indent=1))


def res_dir(smoke):
    return RES / "smoke" if smoke else RES


# ---------------------------------------------------------------- the bundle (real data; heavy imports live here)

def _load_step1():
    spec = importlib.util.spec_from_file_location("run_step1", _T / "20261116_grouping_step1_style/run_step1.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_step1"] = mod
    spec.loader.exec_module(mod)
    return mod


def _subset_scores(s, idx):
    return {c: {d: np.asarray(s[c][d])[idx] for d in DIRECTIONS} for c in CONDITIONS}


def _subset_ctx(ctx, idx):
    from dataclasses import fields
    ep = ctx.pooled
    from src.eval.aspect_episodes import AspectEpisodes
    pooled = AspectEpisodes(ep.aspect_a, ep.aspect_b, *(getattr(ep, f.name)[idx] for f in fields(ep)[2:]))
    return SimpleNamespace(pooled=pooled, anchor_group=ctx.anchor_group[idx], pair_index=ctx.pair_index[idx],
                           parity=ctx.parity[idx], cos=_subset_scores(ctx.cos, idx), n=len(idx), groups=ctx.groups,
                           selection=ctx.selection, in_sel=ctx.in_sel, seed=ctx.seed, smoke=True)


def load_bundle(smoke=False):
    """Everything the readers need, built as step 1 built it, with the rule's regression check (section 5, item 2).
    smoke=True keeps 200 episodes per aspect pair (both parities) and skips only the exact comparison with the stored
    step-1 arrays (the code path still runs)."""
    assert_rule()
    shas = verify_inputs()
    rs1 = _load_step1()
    rto, rsw, rc, rg, n6, df, dc = rs1.rto, rs1.rsw, rs1.rc, rs1.rg, rs1.n6, rs1.df, rs1.dc
    from src.eval.aspect_quick_checks import centered_term

    # local helpers must agree with the originals
    if tuple(rg.POOLED_ORDER) != tuple(POOLED_ORDER) or tuple(rg.PAIRS) != PAIRS:
        raise AssertionError("pair order differs from the evaluation context's")
    probe = {c: {d: np.random.default_rng(0).normal(size=(5, 13)).astype(np.float32) * (1 + (c == "b"))
                 for d in DIRECTIONS} for c in CONDITIONS}
    if not all(np.array_equal(cf_version(probe)[c][d], dc.cf_version(probe)[c][d]) for c in CONDITIONS for d in DIRECTIONS):
        raise AssertionError("cf_version differs from diagnose_counterparts.cf_version")

    t0 = time.time()
    S = rsw.setup()
    ctx_full, cl_full = S.ctx, S.cl
    checks = dict(S.checks)
    stored_L = S.stored["arms"]["L"]
    log("method-A condition-free term T_N1u")
    inp, _, _ = rc.model_inputs(ctx_full, "A3", S.scorer_train, False)
    t_n1u = centered_term(inp, ctx_full.pooled, uniform=True)
    del inp

    log(f"affect L heads with fit_one_head ({n6.HEAD_ROWS} rows)")
    post_L, prov_L = rto.fit_one_head(ctx_full, rto.global_labels(S.partition_L, S.scorer_train, len(ctx_full.groups)),
                                      S.scorer_train, n6.HEAD_ROWS)
    checks["fit_one_head_L_equals_told_oracle_head"] = bool(rto.roundtrip(prov_L) == stored_L["head"])
    if not checks["fit_one_head_L_equals_told_oracle_head"]:
        raise SystemExit(f"affect head {prov_L} differs from told_oracle.json arm L's head {stored_L['head']}")
    z = np.load(INPUT_FILES["step1_heads_style"])
    if not np.array_equal(z["selection"], ctx_full.selection):
        raise AssertionError("step-1 heads were computed on another selection")
    post = {"affect": post_L, "image": S.post_stored["image"], "caption": S.post_stored["caption"]}
    for g, key in (("csd", "style_csd"), ("rand", "style_rand")):
        post[g] = {"img": rs1.full_post(z, f"{key}__img", ctx_full), "txt": rs1.full_post(z, f"{key}__txt", ctx_full)}
    for g in GROUPINGS:
        if not (np.isfinite(post[g]["img"][ctx_full.selection]).all() and np.isfinite(post[g]["txt"][ctx_full.selection]).all()):
            raise AssertionError(f"{g}: non-finite posteriors on selection rows")

    if smoke:
        idx = np.concatenate([np.flatnonzero(ctx_full.pair_index == i)[:SMOKE_PER_PAIR] for i in range(len(PAIRS))])
        ctx = _subset_ctx(ctx_full, idx)
        B, t_n1u = _subset_scores(S.B, idx), _subset_scores(t_n1u, idx)
        pB = per_anchor(B)
    else:
        idx, ctx, B, pB = None, ctx_full, S.B, S.pB
    cl = ctx.anchor_group

    step1_z = np.load(INPUT_FILES["step1_eval_style"])
    step1_post = {a: {STEP1_NAME[h]: post[h] for h in parts} for a, parts in CONFIGS.items()}
    step1_parts = {a: tuple(STEP1_NAME[h] for h in parts) for a, parts in CONFIGS.items()}
    step1_told = {a: {asp: STEP1_NAME[g] for asp, g in TOLD[a].items()} for a in CONFIGS}
    for a in CONFIGS:                                    # the told maps and groupings are step 1's own
        if step1_parts[a] != tuple(rs1.ARMS[a]["parts"]) or step1_told[a] != rs1.ARMS[a]["told"]:
            raise AssertionError(f"{a}: groupings or told mapping differ from run_step1.ARMS")

    Bp, pBp, argmax = {}, {}, {}
    mismatches = []
    for a in CONFIGS:
        t = time.time()
        ev, arr, t6u, picked, _ = rs1.evaluate_general(step1_post[a], step1_parts[a], step1_told[a], ctx, B, pB)
        Bp[a], _ = crossfit_condition_free(ctx.cos, t_n1u, t6u, ctx.parity)
        pBp[a] = per_anchor(Bp[a])
        if not smoke:
            for nm in ("told", "reader"):
                for j, part in enumerate(("fused", "cf")):
                    for m in METRICS:
                        if not np.array_equal(np.asarray(arr[nm][j][m]), step1_z[f"{a}__{nm}__{part}__{m}"]):
                            mismatches.append(f"{a}__{nm}__{part}__{m}")
            for m in METRICS:
                if not np.array_equal(np.asarray(pBp[a][m]), step1_z[f"{a}__Bprime__{m}"]):
                    mismatches.append(f"{a}__Bprime__{m}")
            for c in CONDITIONS:
                if not np.array_equal(picked[c].astype(np.int8), step1_z[f"{a}__reader_pick__{c}"]):
                    mismatches.append(f"{a}__reader_pick__{c}")
        pn, pc = arr["reader"]
        vv, binfo = bar_info(pn, pc, pBp[a], pB, cl, ctx.pair_index)
        argmax[a] = {"fused": pn, "cf": pc, "bar_v": vv, "bar": binfo, "picks": {c: np.asarray(picked[c]) for c in CONDITIONS},
                     "margin_r1": diff3(pn, pc, cl)["r1"]}
        log(f"step-1 arg-max reader on {a} recomputed [{time.time() - t:.0f}s]")
    if not smoke:
        for m in METRICS:
            if not np.array_equal(np.asarray(pB[m]), step1_z[f"B__{m}"]):
                mismatches.append(f"B__{m}")
        if mismatches:
            print(json.dumps(mismatches, indent=1))
            raise SystemExit(f"regression check failed: {len(mismatches)} step-1 arrays differ from step1_eval_style.npz")
        checks["step1_argmax_reader_reproduces_step1_eval_style_exactly(A0,A1,AR; told+reader fused/cf; B'; picks; B)"] = True
        log("regression check passed: step 1 reproduced exactly (A0, A1, AR, B, B')")
    else:
        checks["step1_regression_comparison"] = "skipped (smoke)"
    bundle = SimpleNamespace(ctx=ctx, cl=cl, B=B, pB=pB, post=post, t_n1u=t_n1u, Bp=Bp, pBp=pBp,
                             scorer_train=S.scorer_train, checks=checks, argmax=argmax, parity=ctx.parity,
                             pair_index=ctx.pair_index, input_sha256=shas, smoke=bool(smoke), subset_index=idx,
                             affect_head=prov_L, setup=S, build_s=time.time() - t0)
    return bundle


# ---------------------------------------------------------------- R-a pure pieces (section 4.1)

def sigma_from_agreements(sup, con) -> np.ndarray:
    """(H,) sigma_h = sqrt(mean over episodes of (s2_S,h + s2_C,h) / 4), s2 the sample variance (ddof 1) of the four
    support-pair and the four contrast-pair agreements. sup, con: (E, 4, H)."""
    vs = np.var(np.asarray(sup, dtype=np.float64), axis=1, ddof=1)
    vc = np.var(np.asarray(con, dtype=np.float64), axis=1, ddof=1)
    return np.sqrt(np.mean((vs + vc) / 4.0, axis=0))


def scaled_delta_picks(delta_a, sigma):
    """R-a: scaled Delta^a = Delta^a / sigma (E, H); Delta^b = -Delta^a exactly. Pick = arg max (ties to the first
    grouping); top-two margin = largest minus second-largest scaled Delta. -> (picks, margins, scaled), each a dict
    over the conditions (scaled: (E, H))."""
    sa = np.asarray(delta_a, dtype=np.float64) / np.asarray(sigma, dtype=np.float64)[None, :]
    scaled = {"a": sa, "b": -sa}
    picks = {c: scaled[c].argmax(axis=1) for c in CONDITIONS}
    margins = {c: top_two_margin(scaled[c]) for c in CONDITIONS}
    return picks, margins, scaled


def condition_sets(ep, cond):
    si, st, ci, ct, _ = ep.condition(cond)
    return si, st, ci, ct
