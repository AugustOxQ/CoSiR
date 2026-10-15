"""R-b on the seed-42 development episodes (DECISION_RULE.md §4.2 items 7 and 11, §5 item 2). Every setting is fixed
by the rule; this script chooses none.

  rb_eval.py --config X      X in A1, A0, AR. Loads the bundle (common.load_bundle: B, B', the standard heads and the
                             regression check of §5 item 2), computes the seed-42 features from the standard heads with
                             the same function as the bank (rb_features.both_conditions), P^c(h) = mean of the two
                             half-readers' probabilities, and scores
                               Rb_argmax_X    T^c = s_{pi^c}
                               Rb_expected_X  T^c = sum_h P^c(h) s_h
                             (pick pi^c = arg max P^c, ties to the first grouping; top-two margin = largest minus
                             second-largest P^c), each through common.evaluate and common.save_candidate; then the R-b
                             diagnostics of item 11 -> results/rb_diag_X.json/.txt
  rb_eval.py --summary       after A1 and A0: A1 - A0 under each R-b scoring (paired per-anchor fused R@1 and bar
                             margin, with intervals); the AR rand-pick shares when AR exists -> results/rb_summary.json/.txt

Run (from /project/CoSiR):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261117_reader_fix_csd/rb_eval.py (--config X | --summary) [--smoke]
Non-smoke outputs are never overwritten. --smoke: common's 600-episode subset, smoke readers, results/smoke/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import rb_build as rb  # noqa: E402
import rb_features as rf  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS  # noqa: E402
from src.eval.aspect_quick_checks import aspect_deltas  # noqa: E402

SCORINGS = ("argmax", "expected")


def cand_name(scoring, config):
    return f"Rb_{scoring}_{config}"


def seed42_features(bundle, parts):
    """Features of both conditions from the standard heads (D2), with the feature code of the bank; S, C, the standard
    deviations and the match share are checked against common.pair_agreements / support_argmax_match, Delta against
    aspect_deltas, all exactly."""
    ep, post = bundle.ctx.pooled, bundle.post
    F = rf.both_conditions(post, parts, ep)
    checks = {}
    for c in CONDITIONS:
        si, st, ci, ct = C.condition_sets(ep, c)
        sup, con = C.pair_agreements(post, si, st, ci, ct, parts)
        match = C.support_argmax_match(post, si, st, parts)
        delta = aspect_deltas(post, ep, c, tuple(parts))
        X = F[c]
        ok = True
        for j in range(len(parts)):
            b = 6 * j
            ok &= np.array_equal(X[:, b + 0], sup[:, :, j].mean(axis=1).astype(np.float64))
            ok &= np.array_equal(X[:, b + 1], con[:, :, j].mean(axis=1).astype(np.float64))
            ok &= np.array_equal(X[:, b + 2], delta[:, j].astype(np.float64))
            ok &= np.array_equal(X[:, b + 3], sup[:, :, j].astype(np.float64).std(axis=1, ddof=1))
            ok &= np.array_equal(X[:, b + 4], con[:, :, j].astype(np.float64).std(axis=1, ddof=1))
            ok &= np.array_equal(X[:, b + 5], match[:, :, j].mean(axis=1))
        checks[f"features_{c}_equal_common_and_aspect_deltas"] = bool(ok)
        if not np.isfinite(X).all():
            raise AssertionError(f"seed-42 features of condition {c} are not finite")
    if not all(checks.values()):
        raise SystemExit(f"seed-42 features differ from the shared definitions: {checks}")
    for j in range(len(parts)):
        if not np.array_equal(F["b"][:, 6 * j + 2], -F["a"][:, 6 * j + 2]):
            raise AssertionError("Delta^b must equal -Delta^a exactly (D3)")
    checks["delta_b_equals_minus_delta_a"] = True
    return F, checks


def half_reader_probs(pk, X, n_classes):
    out = []
    for h in pk["halves"]:
        if not np.array_equal(h["model"].classes_, np.arange(n_classes)):
            raise AssertionError("a half-reader's classes differ from the configuration's groupings")
        p = h["model"].predict_proba(h["scaler"].transform(X))
        if not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise AssertionError("half-reader probabilities do not sum to 1")
        out.append(p)
    return out


def shift_report(F42, bank_X, names):
    x42 = np.vstack([F42["a"], F42["b"]])
    s = rf.smd(x42, bank_X)
    rows = {n: (None if not np.isfinite(v) else float(v)) for n, v in zip(names, s)}
    finite = np.abs(s[np.isfinite(s)])
    return {"formula": "(mean on seed 42 - mean on the bank) / sqrt((variance on seed 42 + variance on the bank) / 2), "
                       "variances ddof 1; seed 42: standard heads, both conditions; bank: both halves, both conditions "
                       "(cross-fitted heads)",
            "n_seed42": int(len(x42)), "n_bank": int(len(bank_X)), "smd": rows,
            "max_abs_smd": float(finite.max()) if len(finite) else None,
            "features_abs_smd_above_0.5": [n for n, v in rows.items() if v is not None and abs(v) > 0.5],
            "means": {"seed42": dict(zip(names, x42.mean(axis=0).tolist())),
                      "bank": dict(zip(names, bank_X.mean(axis=0).tolist()))}}


def stage_config(args):
    config = args.config
    parts = C.CONFIGS[config]
    if parts != rb.CONFIGS[config]:
        raise AssertionError("configuration groupings differ between common.py and rb_build.py")
    C.assert_rule()
    out = C.res_dir(args.smoke)
    diag_p = {e: out / f"rb_diag_{config}.{e}" for e in ("json", "txt")}
    if not args.smoke:
        busy = [str(q) for q in diag_p.values() if q.exists()]
        busy += [str(q) for s in SCORINGS for q in C._paths(cand_name(s, config), False)[1].values() if q.exists()]
        if busy:
            raise SystemExit(f"outputs exist ({', '.join(busy)}); refusing to overwrite")
    t0 = time.time()
    pk, rrec, rz = rb.load_readers(config, args.smoke)
    if pk["feature_names"] != rf.feature_names(parts):
        raise AssertionError("the readers were trained on another feature layout")
    bundle = C.load_bundle(smoke=args.smoke)
    ctx, cl = bundle.ctx, bundle.cl
    ep = ctx.pooled

    F42, feat_checks = seed42_features(bundle, parts)
    H = len(parts)
    per_half = {c: half_reader_probs(pk, F42[c], H) for c in CONDITIONS}
    P = {c: rf.average_probs(per_half[c]) for c in CONDITIONS}
    if not all(np.allclose(P[c].sum(axis=1), 1.0, rtol=0, atol=1e-12) for c in CONDITIONS):
        raise AssertionError("averaged probabilities do not sum to 1")
    pm = {c: rf.picks_and_margins(P[c]) for c in CONDITIONS}
    picks = {c: pm[c][0] for c in CONDITIONS}
    margins = {c: pm[c][1] for c in CONDITIONS}
    if not all(np.allclose(margins[c], C.top_two_margin(P[c]), rtol=0, atol=0) for c in CONDITIONS):
        raise AssertionError("top-two margin differs from common.top_two_margin")
    stack = C.grouping_stack(bundle.post, ep, parts)
    terms = {"argmax": C.hard_term(stack, picks), "expected": C.expected_term(stack, P)}

    reader_info = {j: {"chosen_C": rrec["halves"][j]["chosen_C"],
                       "oof_bank_accuracy": rrec["halves"][j]["oof_accuracy_at_chosen_C"]} for j in ("0", "1")}
    common_extra = {"probs_a": P["a"].astype(np.float64), "probs_b": P["b"].astype(np.float64),
                    "anchor_group": np.asarray(cl), "pair_index": np.asarray(ctx.pair_index),
                    "reader": {"config": config, "groupings": list(parts), "half_readers": reader_info,
                               "reader_json_sha256": C.sha_file(rb.reader_paths(config, args.smoke)["json"]),
                               "reader_pkl_sha256": rrec["provenance"]["pkl_sha256"],
                               "pick": "arg max_h P^c(h), ties to the first grouping in configuration order",
                               "margin": "largest minus second-largest P^c(h)"},
                    "feature_checks": feat_checks, "bundle_checks": bundle.checks}
    summaries = {}
    for s in SCORINGS:
        name = cand_name(s, config)
        summary, arrays = C.evaluate(bundle, config, terms[s], picks, name)
        summary["smoke"] = bool(args.smoke)
        summary["scoring"] = ("R-b arg-max: T^c = s_{pi^c}" if s == "argmax"
                              else "R-b expected: T^c = sum_h P^c(h) s_h")
        if config == "AR":
            summary["AR_check_rand_share"] = {
                "overall": summary["pick_share"]["overall"]["rand"],
                "per_pair_condition": {p: {c: v[c]["rand"] for c in CONDITIONS}
                                       for p, v in summary["pick_share"]["per_pair_condition"].items()}}
        C.save_candidate(name, summary, arrays, terms[s], picks, margins, extra={**common_extra, "scoring": s},
                         smoke=args.smoke)
        summaries[s] = summary
        C.log(f"{name} saved [{time.time() - t0:.0f}s]")

    # ---------------------------------------------------------- R-b diagnostics (item 11; enter no rule)
    names = rf.feature_names(parts)
    bank_X = np.vstack([rz["half0__X"], rz["half1__X"]])
    oof = np.vstack([rz["half0__oof_proba"], rz["half1__oof_proba"]])
    top42 = np.concatenate([P[c].max(axis=1) for c in CONDITIONS])
    diag = {"config": config, "groupings": list(parts), "smoke": bool(args.smoke),
            "a_bank_accuracy": {j: {"oof_accuracy_at_chosen_C": rrec["halves"][j]["oof_accuracy_at_chosen_C"],
                                    "chosen_C": rrec["halves"][j]["chosen_C"],
                                    "cv_mean_log_loss": {str(r["C"]): r["mean_log_loss"] for r in rrec["halves"][j]["cv_table"]},
                                    "convergence_warnings_cv": {str(r["C"]): r["convergence_warnings"]
                                                                for r in rrec["halves"][j]["cv_table"]},
                                    "convergence_warnings_refit": rrec["halves"][j]["refit_convergence_warnings"]}
                                for j in ("0", "1")},
            "chance": 100.0 / H,
            "b_pick_accuracy_seed42": summaries["argmax"]["pick_accuracy"],
            "pick_share_seed42": summaries["argmax"]["pick_share"],
            "c_shift_report": shift_report(F42, bank_X, names),
            "top_probability": {"bank_oof_pooled_halves": rf.distribution(oof.max(axis=1)),
                                "seed42_both_conditions": rf.distribution(top42),
                                "seed42_per_condition": {c: rf.distribution(P[c].max(axis=1)) for c in CONDITIONS}},
            "half_reader_agreement_seed42": {c: 100 * float(np.mean(per_half[c][0].argmax(1) == per_half[c][1].argmax(1)))
                                             for c in CONDITIONS},
            "candidates": {s: {"name": cand_name(s, config), "bar_r1": summaries[s]["bar"]["r1"],
                               "bar_comparator": summaries[s]["bar"]["comparator"],
                               "gain_statistic": summaries[s]["gain_statistic"],
                               "margin_r1": summaries[s]["margin"]["r1"]} for s in SCORINGS},
            "note": "R-b diagnostics (DECISION_RULE.md §4.2 item 11) enter no rule; there is no kill for R-b"}
    if config == "AR":
        ref = bundle.argmax["AR"]["picks"]
        diag["AR_check"] = {"rand_share_Rb": summaries["argmax"]["AR_check_rand_share"],
                            "rand_share_step1_argmax": C.pick_shares(ref, parts, ctx.pair_index)["overall"]["rand"],
                            "rand_share_step1_argmax_per_pair_condition": {
                                p: {c: v[c]["rand"] for c in CONDITIONS}
                                for p, v in C.pick_shares(ref, parts, ctx.pair_index)["per_pair_condition"].items()}}
    diag["runtime_s"] = round(time.time() - t0, 1)
    write_once(diag_p, diag, diag_text, args.smoke)


def write_once(p, rec, text_fn, smoke):
    C.write_json_once(p["json"], rec, smoke)
    rec = json.loads(p["json"].read_text())
    txt = text_fn(rec)
    p["txt"].write_text(txt + "\n")
    print(txt, flush=True)


def _ci(x):
    return f"{x['point']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def diag_text(r):
    L = [f"R-b diagnostics, {r['config']} ({', '.join(r['groupings'])}){' [SMOKE: not results]' if r['smoke'] else ''}; "
         "enter no rule (DECISION_RULE.md §4.2 item 11)"]
    for j, a in r["a_bank_accuracy"].items():
        L.append(f"  (a) half {j}: out-of-fold bank accuracy {a['oof_accuracy_at_chosen_C']:.2f}% at C "
                 f"{a['chosen_C']} (chance {r['chance']:.1f}); refit warnings {a['convergence_warnings_refit']}")
    pa = r["b_pick_accuracy_seed42"]
    L.append(f"  (b) seed-42 pick accuracy {_ci(pa['correct_share'])} (chance {pa['chance']:.1f}); per pair a/b: "
             + "; ".join(f"{p} {v['a']:.1f}/{v['b']:.1f}" for p, v in pa["per_pair_condition"].items()))
    L.append("      pick shares overall: " + ", ".join(f"{h} {v:.1f}" for h, v in r["pick_share_seed42"]["overall"].items()))
    sr = r["c_shift_report"]
    L.append(f"  (c) shift report: SMD per feature (seed 42 n={sr['n_seed42']} vs bank n={sr['n_bank']}); max |SMD| "
             f"{sr['max_abs_smd']}; |SMD| > 0.5: {sr['features_abs_smd_above_0.5']}")
    for h in r["groupings"]:
        L.append(f"      {h:8s} " + "  ".join(
            f"{f}={'null' if sr['smd'][f'{h}__{f}'] is None else format(sr['smd'][f'{h}__{f}'], '+.2f')}"
            for f in rf.FEATURES))
    for k, d in (("bank OOF (pooled halves)", r["top_probability"]["bank_oof_pooled_halves"]),
                 ("seed 42 (both conditions)", r["top_probability"]["seed42_both_conditions"])):
        L.append(f"      top probability, {k}: mean {d['mean']:.3f}; deciles "
                 + " ".join(f"{v:.3f}" for v in d["deciles"].values()))
    L.append(f"  half-reader pick agreement on seed 42 (%): a {r['half_reader_agreement_seed42']['a']:.1f}, "
             f"b {r['half_reader_agreement_seed42']['b']:.1f}")
    for s, c in r["candidates"].items():
        L.append(f"  {c['name']}: bar margin {_ci(c['bar_r1'])} vs {c['bar_comparator']}; gain statistic "
                 f"{_ci(c['gain_statistic'])}; margin R@1 {_ci(c['margin_r1'])}")
    if "AR_check" in r:
        a = r["AR_check"]
        L.append(f"  AR check: share of picks to rand, R-b {a['rand_share_Rb']['overall']:.1f}% vs step-1 arg-max "
                 f"{a['rand_share_step1_argmax']:.1f}%")
    L.append(f"runtime {r['runtime_s']:.0f}s")
    return "\n".join(L)


def stage_summary(args):
    C.assert_rule()
    out = C.res_dir(args.smoke)
    p = {e: out / f"rb_summary.{e}" for e in ("json", "txt")}
    if not args.smoke and any(q.exists() for q in p.values()):
        raise SystemExit(f"rb_summary exists in {out}; refusing to overwrite")
    res = {"smoke": bool(args.smoke), "A1_minus_A0": {}, "descriptive": "DECISION_RULE.md §5 item 2: A1 - A0 under "
           "each reader, paired per anchor; descriptive, decides nothing"}
    for s in SCORINGS:
        recs = {}
        for cfg in ("A1", "A0"):
            rec, z = C.load_candidate(cand_name(s, cfg), args.smoke)
            recs[cfg] = (rec, z)
        z1, z0 = recs["A1"][1], recs["A0"][1]
        cl = z1["extra__anchor_group"]
        if not (np.array_equal(cl, z0["extra__anchor_group"])
                and np.array_equal(z1["extra__pair_index"], z0["extra__pair_index"])):
            raise AssertionError("A1 and A0 candidates were evaluated on different episodes")
        d_r1 = np.asarray(z1["fused__r1"], np.float64) - np.asarray(z0["fused__r1"], np.float64)
        d_bar = np.asarray(z1["bar_v"], np.float64) - np.asarray(z0["bar_v"], np.float64)
        res["A1_minus_A0"][s] = {
            "names": [cand_name(s, "A1"), cand_name(s, "A0")],
            "fused_r1": C.point_ci(d_r1, cl), "bar_margin_r1": C.point_ci(d_bar, cl),
            "bar_comparators": {cfg: recs[cfg][0]["bar"]["comparator"] for cfg in recs},
            "fused_r1_means": {cfg: recs[cfg][0]["r1_means"]["fused"] for cfg in recs},
            "bar_margins": {cfg: recs[cfg][0]["bar"]["r1"] for cfg in recs}}
    ar = {}
    for s in SCORINGS:
        _, cp = C._paths(cand_name(s, "AR"), args.smoke)
        if cp["json"].exists():
            rec, _ = C.load_candidate(cand_name(s, "AR"), args.smoke)
            ar[s] = {"rand_share": rec["AR_check_rand_share"], "margin_r1": rec["margin"]["r1"],
                     "bar_r1": rec["bar"]["r1"], "bar_comparator": rec["bar"]["comparator"]}
    res["AR_check"] = ar if ar else "AR not run yet"
    write_once(p, res, summary_text, args.smoke)


def summary_text(r):
    L = [f"R-b: A1 - A0 under each scoring (paired per anchor, pp, 95% painting-bootstrap intervals; descriptive)"
         f"{' [SMOKE: not results]' if r['smoke'] else ''}"]
    for s, v in r["A1_minus_A0"].items():
        L.append(f"  {s:8s}: fused R@1 {_ci(v['fused_r1'])}; bar margin {_ci(v['bar_margin_r1'])} (comparators "
                 f"A1 {v['bar_comparators']['A1']}, A0 {v['bar_comparators']['A0']})")
    if isinstance(r["AR_check"], dict):
        for s, v in r["AR_check"].items():
            L.append(f"  AR {s}: picks to rand {v['rand_share']['overall']:.1f}%; margin {_ci(v['margin_r1'])}; bar "
                     f"margin {_ci(v['bar_r1'])} vs {v['bar_comparator']}")
    else:
        L.append(f"  AR: {r['AR_check']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", choices=tuple(C.CONFIGS))
    g.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    C.assert_rule()
    if args.summary:
        stage_summary(args)
    else:
        stage_config(args)


if __name__ == "__main__":
    main()
