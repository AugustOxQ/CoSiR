"""Quick checks D0, N1 and N2 on the seed-42 development episodes. The rules are DECISION_RULE.md in this folder
(committed in 7e50f18 before any scorer existed); this script applies them mechanically. CPU only.

Real run:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
           python src/test/20261108_new_method_quick_checks/run_checks.py         -> results/ (refuses to overwrite)
Smoke run: ... run_checks.py --smoke    -> results/smoke/ (E1's smoke episodes; A1's smoke checkpoint stands in for
           A3, L3 and LT; probes fitted on 3,000 rows)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src/test/20261101_aspect_factor_gonogo"))
import run_gonogo as rg  # noqa: E402  (puts the repo root on sys.path)

from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import METRICS, cluster_bootstrap, compare, per_anchor  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import (ASPECTS, CONFIG_ORDER, both_in_topk, cascade_scores,  # noqa: E402
                                          centered_term, code_scale, config_passes, d0_reading, decision_row,
                                          inferred_scores, kissme_diag_term, n1_stop, n2_reading, told_scores)
from src.eval.aspect_scorers import EvalInputs, agreement_term  # noqa: E402
from src.train.train_factors import encode_rows, load_factor_checkpoint  # noqa: E402

SEED = 42
KS = (2, 3, 5)
PROBE_ROWS, SMOKE_PROBE_ROWS, PROBE_SEED = 60_000, 3_000, 0
DECIDING = ("A3", "C0", "SE")                 # may pass DECISION_RULE.md §4
DIAGNOSTIC = ("L3", "LT")                     # label-trained: readings only
NESTED = ("agree", "N1", "kissme")            # conditioned terms inside the nested score
A_PRIME = ROOT / "src/test/20261105_method_repair_diagnostics"
A3_SHA = "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2"
CODE_SHA = {"C0": "71559d058314a45b863d0f47500bd06ee3ff56623142806c02841099e3d7c43c",
            "SE": "845d6cd330f84adfdacd0ab98db40d54289302e9e39f827248f1df441e2a78e1"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def rel(p):
    return str(Path(p).relative_to(ROOT))


def point_ci(values, clusters):
    """Pooled mean (pp) and painting-clustered 95% interval of one per-anchor array."""
    r = cluster_bootstrap(values, clusters)
    return {"point": 100 * r["point"], "ci95": [100 * c for c in r["ci95"]]}


def assert_finite_scores(scores, what):
    for c in scores:
        for d in scores[c]:
            bad = int((~np.isfinite(np.asarray(scores[c][d]))).any(axis=1).sum())
            if bad:
                raise AssertionError(f"{what}: {bad} episodes with a non-finite score ({c}, {d})")


def flat_share(scores):
    """Share (pp) of rankings whose score row is constant (every candidate tied, which per_anchor counts as a miss)."""
    return 100 * float(np.mean([np.ptp(np.asarray(scores[c][d], np.float64), axis=1) == 0
                                for c in scores for d in scores[c]]))


class Report:
    """Summaries of per-anchor arrays against the comparators shared by every scorer of one episode set."""

    def __init__(self, ctx, cos_pa, rca_pa):
        self.ctx, self.cl, self.cos, self.rca = ctx, ctx.anchor_group, cos_pa, rca_pa

    def describe(self, pa):
        return {"summary": self.ctx.summary(pa), "either": point_ci(pa["r1"] + pa["other"], self.cl)}

    def paired(self, pa, pb):
        return {m: compare(pa, pb, self.cl, m) for m in ("r1", "gain")}

    def full(self, pa, control=None):
        out = {**self.describe(pa), "vs_cosine": self.paired(pa, self.cos), "vs_rca": self.paired(pa, self.rca)}
        if control is not None:
            out["vs_control"] = self.paired(pa, control)
        return out


def fit_probes(ctx, labels, scorer_train, n_rows):
    """D0's representation (spec §4.1): one logistic regression per aspect and modality on unit-normalised CLIP
    features of n_rows scorer-train rows with their labels (rows labelled -1 left out); posteriors on selection rows,
    NaN elsewhere."""
    tr = np.random.default_rng(PROBE_SEED).choice(scorer_train, n_rows, replace=False)
    sel = ctx.selection
    feats = {"img": ctx.data.img_features, "txt": ctx.data.txt_features}
    post, prov = {}, {"draw_rows_sha256": rg.sha_array(np.sort(tr)), "n_draw": int(n_rows), "seed": PROBE_SEED}
    for h in ASPECTS:
        lab = np.asarray(labels[h])
        fit = tr[lab[tr] >= 0]
        clfs = {m: LogisticRegression(C=1.0, max_iter=300).fit(unit(F[fit]), lab[fit]) for m, F in feats.items()}
        if not np.array_equal(clfs["img"].classes_, clfs["txt"].classes_):
            raise AssertionError(f"{h}: image and caption probes have different classes")
        labelled = sel[lab[sel] >= 0]
        post[h] = {}
        for m, F in feats.items():
            full = np.full((len(ctx.groups), len(clfs[m].classes_)), np.nan, dtype=np.float32)
            full[sel] = clfs[m].predict_proba(unit(F[sel]))
            if not (np.isfinite(full[sel]).all() and np.isnan(full[~ctx.in_sel]).all()):
                raise AssertionError(f"{h}/{m}: posteriors must be finite on selection rows and NaN elsewhere")
            post[h][m] = full
        prov[h] = {"n_fit": int(len(fit)), "classes": clfs["img"].classes_.tolist(),
                   "selection_accuracy": {m: 100 * float(clfs[m].score(unit(F[labelled]), lab[labelled]))
                                          for m, F in feats.items()}}
        log(f"probe {h}: fit on {len(fit)} rows")
    return post, prov


def model_inputs(ctx, name, scorer_train, smoke):
    """(EvalInputs with selection-masked codes, per-factor scale from scorer-train codes, provenance)."""
    if name in CODE_SHA:
        path = rg.E1 / f"codes_{name}.npz"
        sha = rg.sha_file(path)
        if sha != CODE_SHA[name]:
            raise AssertionError(f"{path}: SHA-256 {sha} differs from DECISION_RULE.md")
        z = np.load(path)
        ic, tc = ctx.masked(z["img"]), ctx.masked(z["txt"])
        train_img, train_txt = z["img"][scorer_train], z["txt"][scorer_train]
        prov = {"codes": rel(path), "sha256": sha}
    else:
        if smoke:
            ckpt = rg.checkpoint_path(name, SEED, True)            # A1's smoke checkpoint stands in
        elif name == "A3":
            ckpt = rg.checkpoint_path("A3", SEED, False)
        else:
            ckpt = A_PRIME / "checkpoints" / f"{name}_seed{SEED}.pt"
        sha = rg.sha_file(ckpt)
        if not smoke and name == "A3":
            pick = json.loads((rg.HERE / "results" / "picked.json").read_text())
            if not (pick["run"] == "A3" and sha == pick["checkpoint_sha256"] == A3_SHA):
                raise AssertionError("A3 checkpoint is not E3's pick")
        elif not smoke:
            listed = json.loads((A_PRIME / "results" / "label_checkpoints.json").read_text())
            if listed.get(f"{name}_seed{SEED}") != sha:
                raise AssertionError(f"{name}: SHA-256 not in label_checkpoints.json")
        ic, tc = ctx.encode(ckpt)
        model, _ = load_factor_checkpoint(ckpt, device="cpu")
        train_img, train_txt = encode_rows(model, ctx.data.img_features, ctx.data.txt_features, rows=scorer_train,
                                           device="cpu")
        prov = {"checkpoint": rel(ckpt), "sha256": sha}
    return EvalInputs(ctx.img, ctx.txt, ic, tc), code_scale(train_img, train_txt), prov


def summary_text(r):
    def c(x):
        return f"{x['point']:6.2f} [{x['ci95'][0]:6.2f}, {x['ci95'][1]:6.2f}]"

    p = r["provenance"]
    lines = [f"Quick checks, seed {SEED}, n={p['n_episodes']}{' (SMOKE)' if p['smoke'] else ''}",
             f"cosine R@1 {c(r['cosine']['summary']['r1'])} either {r['cosine']['either']['point']:.2f}; "
             f"RCA R@1 {c(r['rca']['summary']['r1'])} gain {c(r['rca']['summary']['gain'])}", "D0 (label probes):"]
    for v in ("told", "hard", "soft"):
        e = r["d0"][v]
        lines.append(f"  {v:5s} R@1 {c(e['summary']['r1'])} gain {c(e['summary']['gain'])} "
                     f"either {e['either']['point']:6.2f}")
    d = r["d0"]["reading"]
    lines.append(f"  hard-pick accuracy {r['d0']['hard_pick_accuracy']['pooled']:.1f}%; soft fallback "
                 f"{r['d0']['soft_fallback_share']:.2f}%; variant {d['variant']} gain {d['gain']:.2f} vs threshold "
                 f"{d['threshold']:.2f} -> {d['reading']}")
    lines.append("Factor codes: term only (R@1, gain, either, flat %), control, nested vs control:")
    for name, e in r["factors"].items():
        for t, x in e["term_only"].items():
            lines.append(f"  {name:3s} term {t:10s} R@1 {x['summary']['r1']['point']:6.2f} gain "
                         f"{x['summary']['gain']['point']:6.2f} either {x['either']['point']:6.2f} "
                         f"flat {x['flat_share']:5.2f}")
        lines.append(f"  {name:3s} N1 stop: {e['n1_stop']}")
        lines.append(f"  {name:3s} control R@1 {e['control']['summary']['r1']['point']:6.2f} "
                     f"either {e['control']['either']['point']:6.2f}")
        for t, x in e["nested"].items():
            lines.append(f"  {name:3s} nested {t:7s} R@1 {c(x['summary']['r1'])} gain {c(x['summary']['gain'])} "
                         f"vs control R@1 {c(x['vs_control']['r1'])} picks {x['picks']}")
    lines.append("N2 on A3:")
    for k in KS:
        lines.append(f"  k={k} both-in-top-k {c(r['n2'][f'both_in_top{k}'])}")
        for rr in r["n2"]["rerankers"]:
            x = r["n2"][f"{k}-{rr}"]
            lines.append(f"    {rr:5s} R@1 {c(x['summary']['r1'])} gain {c(x['summary']['gain'])} either "
                         f"{x['either']['point']:6.2f} vs control R@1 {c(x['vs_control']['r1'])} {x['reading']}")
    lines.append("Configurations (pass = R@1 and gain lower bounds against the own control above 0):")
    for x in r["configs"]:
        lines.append(f"  {x['name']:14s} m_R {x['m_r']:6.2f} {x['r1_ci95']}  m_g {x['m_g']:6.2f} {x['gain_ci95']}  "
                     f"pass {x['passes']}")
    dec = r["decision"]
    lines.append(f"DECISION: row {dec['row']} ({dec['config']}): {dec['next']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    res = HERE / "results" / ("smoke" if smoke else "")
    names = ("checks_seed42.json", "per_anchor_checks_seed42.npz", "checks_seed42.txt", "decision.json")
    if not smoke and any((res / n).exists() for n in names):
        raise SystemExit(f"results exist in {res}; the checks run once. Refusing to overwrite.")
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ctx = rg.EvalContext(SEED, smoke)
    ep, cl = ctx.pooled, ctx.anchor_group
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    labels = artelingo_aspect_labels(ctx.data)
    t9 = np.load(rg.folders(smoke)["e1"] / "per_anchor_seed42.npz")
    cos_pa = per_anchor(ctx.cos)
    if not all(np.array_equal(cos_pa[m], t9[f"cosine__{m}"]) for m in METRICS):
        raise AssertionError("episodes misaligned with E1's seed-42 arrays")
    rca_pa = {m: t9[f"rca__{m}"].astype(np.float64) for m in METRICS}
    rep = Report(ctx, cos_pa, rca_pa)
    arrays = {"anchor_group": cl, "pair_index": ctx.pair_index}

    def keep(prefix, pa):
        for m in METRICS:
            arrays[f"{prefix}__{m}"] = np.asarray(pa[m])

    keep("cosine", cos_pa)
    rule = HERE / "DECISION_RULE.md"
    result = {"provenance": {"decision_rule": {"path": rel(rule), "sha256": rg.sha_file(rule), "commit": "7e50f18"},
                             "script_sha256": rg.sha_file(Path(__file__)),
                             "module_sha256": rg.sha_file(ROOT / "src/eval/aspect_quick_checks.py"),
                             "episodes_sha256": ctx.shas, "n_episodes": int(ctx.n),
                             "n_clusters": int(len(np.unique(cl))), "sklearn": sklearn.__version__, "smoke": smoke},
              "cosine": rep.describe(cos_pa), "rca": rep.describe(rca_pa)}
    either_cos = result["cosine"]["either"]["point"]
    log(f"context ready: {ctx.n} episodes")

    # ---------------------------------------------------------------- D0
    post, probe_prov = fit_probes(ctx, labels, scorer_train, SMOKE_PROBE_ROWS if smoke else PROBE_ROWS)
    aspect_a = np.array([ASPECTS.index(rg.PAIRS[i][0]) for i in ctx.pair_index])
    aspect_b = np.array([ASPECTS.index(rg.PAIRS[i][1]) for i in ctx.pair_index])
    hard, hard_info = inferred_scores(post, ep, "hard")
    soft, soft_info = inferred_scores(post, ep, "soft")
    d0_pa = {}
    for name, s in (("told", told_scores(post, ep, aspect_a, aspect_b)), ("hard", hard), ("soft", soft)):
        assert_finite_scores(s, f"D0 {name}")
        d0_pa[name] = per_anchor(s)
        keep(f"d0_{name}", d0_pa[name])
    d0 = {name: {**rep.full(pa), "per_pair": ctx.per_pair(pa)} for name, pa in d0_pa.items()}
    truth = {"a": aspect_a, "b": aspect_b}
    picked = {c: hard_info[c]["weights"].argmax(axis=1) for c in truth}
    acc = {c: 100 * float(np.mean(picked[c] == truth[c])) for c in truth}
    acc["pooled"] = 0.5 * (acc["a"] + acc["b"])
    d0["hard_pick_accuracy"] = acc
    d0["hard_pick_accuracy_per_pair"] = {
        rg.POOLED_ORDER[i]: 100 * float(np.mean([np.mean(picked[c][ctx.pair_index == i] == truth[c][ctx.pair_index == i])
                                                 for c in truth])) for i in range(len(rg.PAIRS))}
    d0["soft_fallback_share"] = 100 * float(np.mean([soft_info[c]["fallback"] for c in truth]))
    reading = d0_reading(d0["told"]["summary"]["gain"], {v: d0[v]["summary"]["gain"]["point"] for v in ("hard", "soft")})
    var = d0_pa[reading["variant"]]
    d0["gain_minus_half_told"] = point_ci(var["gain"] - 0.5 * d0_pa["told"]["gain"], cl)
    d0["r1_analogue"] = point_ci((var["r1"] - cos_pa["r1"]) - 0.5 * (d0_pa["told"]["r1"] - cos_pa["r1"]), cl)
    d0["reading"], d0["probes"] = reading, probe_prov
    result["d0"] = d0
    log(f"D0 done: {reading['reading']}")

    # ---------------------------------------------------------------- N1, the current rule, diagonal KISSME
    factors, n1_stops, controls = {}, {}, {}
    pilot = None if smoke else np.load(A_PRIME / "results" / "per_anchor_pilot_seed42.npz")
    terms_a3 = None
    for name in (*DECIDING, *DIAGNOSTIC):
        inp, scale, prov = model_inputs(ctx, name, scorer_train, smoke)
        terms = {"agree": agreement_term(inp, ep), "N1": centered_term(inp, ep),
                 "N1_uniform": centered_term(inp, ep, uniform=True), "kissme": kissme_diag_term(inp, ep, scale),
                 "uniform": agreement_term(inp, ep, uniform=True)}
        entry = {"provenance": prov, "term_only": {}, "nested": {}}
        tpa = {}
        for t, s in terms.items():
            assert_finite_scores(s, f"{name} {t}")
            tpa[t] = per_anchor(s)
            keep(f"{name}__term_{t}", tpa[t])
            entry["term_only"][t] = {**rep.describe(tpa[t]), "flat_share": flat_share(s)}
        entry["n1_minus_current_gain"] = compare(tpa["N1"], tpa["agree"], cl, "gain")
        entry["n1_stop"] = n1_stop(entry["n1_minus_current_gain"]["point"], entry["term_only"]["N1"]["either"]["point"],
                                   either_cos)
        n1_stops[name] = entry["n1_stop"]
        for t in NESTED:
            nested, control, picks = crossfit_nested(ctx.cos, terms["uniform"], terms[t], ctx.parity)
            pn, pc = per_anchor(nested), per_anchor(control)
            if name not in controls:
                controls[name] = (control, pc)
                entry["control"] = rep.describe(pc)
                keep(f"{name}__control", pc)
            elif not all(np.array_equal(pc[m], controls[name][1][m]) for m in METRICS):
                raise AssertionError(f"{name}: the nested uniform control must not depend on the conditioned term")
            keep(f"{name}__nested_{t}", pn)
            entry["nested"][t] = {**rep.full(pn, control=pc), "picks": picks}
            if pilot is not None and t == "agree" and name in DECIDING:
                for kind, pa in (("nested", pn), ("control", pc)):
                    if not all(np.array_equal(pa[m], pilot[f"{name}__{kind}__{m}"]) for m in METRICS):
                        raise AssertionError(f"{name} {kind}: differs from the A′ pilot's stored arrays")
        if name == "A3":
            terms_a3 = terms
        factors[name] = entry
        log(f"{name} done")
    result["factors"] = factors

    # ---------------------------------------------------------------- N2 on A3
    control_a3, pc_a3 = controls["A3"]
    rerankers = {"agree": terms_a3["agree"]}
    if not n1_stops["A3"]["stop"]:
        rerankers["N1"] = terms_a3["N1"]
    one = per_anchor(cascade_scores(control_a3, terms_a3["agree"], 1))
    if not all(np.array_equal(one[m], pc_a3[m]) for m in METRICS):
        raise AssertionError("the k = 1 cascade must equal its control")
    n2 = {"rerankers": list(rerankers)}
    for k in KS:
        both = both_in_topk(control_a3, k)
        arrays[f"n2_both_top{k}"] = both
        n2[f"both_in_top{k}"] = point_ci(both, cl)
        for rr, term in rerankers.items():
            casc = cascade_scores(control_a3, term, k)
            assert_finite_scores(casc, f"N2 k={k} {rr}")
            pa = per_anchor(casc)
            keep(f"n2_{k}_{rr}", pa)
            entry = rep.full(pa, control=pc_a3)
            entry["reading"] = n2_reading(n2[f"both_in_top{k}"]["point"], entry["vs_control"]["gain"])
            n2[f"{k}-{rr}"] = entry
    result["n2"] = n2
    log("N2 done")

    # ---------------------------------------------------------------- decision (DECISION_RULE.md §4)
    configs = []
    for cname in CONFIG_ORDER:
        kind, a, b = cname.split("-")
        if kind == "N1":
            vc = factors[b]["nested"]["N1"]["vs_control"]
        elif f"{a}-{b}" in n2:
            vc = n2[f"{a}-{b}"]["vs_control"]
        else:
            continue                                   # N2 with N1's term: not computed because N1 stopped on A3
        configs.append({"name": cname, "passes": config_passes(vc["r1"], vc["gain"]), "m_r": vc["r1"]["point"],
                        "m_g": vc["gain"]["point"], "r1_ci95": vc["r1"]["ci95"], "gain_ci95": vc["gain"]["ci95"]})
    decision = decision_row(configs, reading["reading"])
    result["configs"], result["decision"] = configs, decision
    rg.assert_finite_tree(result)
    record = {"d0": reading, "configs": configs, "n1_stop": n1_stops,
              "n2_readings": {k: v["reading"] for k, v in n2.items() if isinstance(v, dict) and "reading" in v},
              "decision": decision, "smoke": smoke}
    (res / "checks_seed42.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(res / "per_anchor_checks_seed42.npz", **arrays)
    (res / "decision.json").write_text(json.dumps(record, indent=1))
    text = summary_text(result)
    (res / "checks_seed42.txt").write_text(text + "\n")
    print(text)
    log(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
