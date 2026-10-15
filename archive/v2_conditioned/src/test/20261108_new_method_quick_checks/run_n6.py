"""N6 (ADDENDUM_2_N6.md in this folder, committed before this code): cross-modal heads on E2's label-free partitions,
D0's hard reader over the partitions, and the nested score against its own matched condition-free control. CPU only.

Dev:   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
       python src/test/20261108_new_method_quick_checks/run_n6.py --stage dev     -> results/n6_seed42.*
Test:  ... run_n6.py --stage test    -> results/n6_test.* (exits unless the seed-42 check passed)
Smoke: add --smoke (results/smoke/; E1's smoke episodes, seeds 42 and 43 stand in for the test seeds, heads fitted on
       3,000 rows; the smoke test stage runs whatever the smoke dev stage found)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_checks as rc  # noqa: E402  (puts run_gonogo and the repo root on sys.path)

from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import METRICS, compare, per_anchor, summarize  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import (config_passes, go_verdict, inferred_scores,  # noqa: E402
                                          uniform_probe_scores)

rg = rc.rg
PARTS = ("affect", "image", "caption")
PARTITIONS = rg.E2 / "partitions.npz"
PARTITIONS_SHA = "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa"
DEV_SEED, TEST_SEEDS, SMOKE_TEST_SEEDS = 42, (45, 47, 48), (42, 43)
HEAD_ROWS, SMOKE_HEAD_ROWS, CHECK_ROWS = 60_000, 3_000, 10_000
SCORERS = ("config", "control", "cosine", "rca")


def partition_labels(groups, scorer_train):
    """Global-row cluster ids of the three partitions (-1 outside scorer-train); alignment with E2 asserted."""
    if rg.sha_file(PARTITIONS) != PARTITIONS_SHA:
        raise AssertionError(f"{PARTITIONS}: SHA-256 differs from ADDENDUM_2_N6.md")
    z = np.load(PARTITIONS)
    if not np.array_equal(z["local_groups"], np.unique(groups[scorer_train], return_inverse=True)[1]):
        raise AssertionError("E2 partitions are not aligned with artelingo_splits().scorer_train")
    out = {}
    for h in PARTS:
        lab = np.full(len(groups), -1, dtype=np.int64)
        lab[scorer_train] = z[h]
        out[h] = lab
    return out


def fit_heads(ctx, labels, scorer_train, n_rows):
    """One logistic regression per partition and modality on D0's row draw; posteriors on selection rows (NaN
    elsewhere), plus held-out accuracy on 10,000 other scorer-train rows."""
    draw = np.random.default_rng(rc.PROBE_SEED).choice(scorer_train, n_rows, replace=False)
    rest = np.setdiff1d(scorer_train, draw)
    check = np.random.default_rng(1).choice(rest, min(CHECK_ROWS, len(rest)), replace=False)
    feats = {"img": ctx.data.img_features, "txt": ctx.data.txt_features}
    sel = ctx.selection
    post, prov = {}, {"draw_rows_sha256": rg.sha_array(np.sort(draw)), "n_draw": int(n_rows)}
    for h in PARTS:
        lab = labels[h]
        clfs = {m: LogisticRegression(C=1.0, max_iter=300).fit(rc.unit(F[draw]), lab[draw]) for m, F in feats.items()}
        if not np.array_equal(clfs["img"].classes_, clfs["txt"].classes_):
            raise AssertionError(f"{h}: image and caption heads have different classes")
        post[h] = {}
        for m, F in feats.items():
            full = np.full((len(ctx.groups), len(clfs[m].classes_)), np.nan, dtype=np.float32)
            full[sel] = clfs[m].predict_proba(rc.unit(F[sel]))
            if not (np.isfinite(full[sel]).all() and np.isnan(full[~ctx.in_sel]).all()):
                raise AssertionError(f"{h}/{m}: posteriors must be finite on selection rows and NaN elsewhere")
            post[h][m] = full
        prov[h] = {"n_classes": int(len(clfs["img"].classes_)),
                   "heldout_accuracy": {m: 100 * float(clfs[m].score(rc.unit(F[check]), lab[check]))
                                        for m, F in feats.items()}}
        rc.log(f"head {h} done")
    return post, prov


def load_posteriors(path, ctx):
    z = np.load(path)
    if not np.array_equal(z["selection"], ctx.selection):
        raise AssertionError("saved posteriors were computed on another selection row set")
    post = {}
    for h in PARTS:
        post[h] = {}
        for m in ("img", "txt"):
            full = np.full((len(ctx.groups), z[f"{h}__{m}"].shape[1]), np.nan, dtype=np.float32)
            full[ctx.selection] = z[f"{h}__{m}"]
            post[h][m] = full
    return post


def n6_terms(post, ep):
    """(T_6 hard, T_6soft, T_6u, hard info, soft info)."""
    hard, info = inferred_scores(post, ep, "hard", PARTS)
    soft, soft_info = inferred_scores(post, ep, "soft", PARTS)
    return hard, soft, uniform_probe_scores(post, ep, PARTS), info, soft_info


def e1_arrays(ctx, smoke, seed):
    """Cosine (recomputed, asserted equal to E1's) and RCA per-anchor arrays of one seed."""
    t9 = np.load(rg.folders(smoke)["e1"] / f"per_anchor_seed{seed}.npz")
    cos_pa = per_anchor(ctx.cos)
    if not (all(np.array_equal(cos_pa[m], t9[f"cosine__{m}"]) for m in METRICS)
            and np.array_equal(t9["anchor_group"], ctx.anchor_group)
            and np.array_equal(t9["pair_index"], ctx.pair_index)):
        raise AssertionError(f"seed {seed}: episodes misaligned with E1's per-anchor arrays")
    return cos_pa, {m: t9[f"rca__{m}"].astype(np.float64) for m in METRICS}


def dev_stage(smoke, res):
    ctx = rg.EvalContext(DEV_SEED, smoke)
    cl, ep = ctx.anchor_group, ctx.pooled
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    cos_pa, rca_pa = e1_arrays(ctx, smoke, DEV_SEED)
    rep = rc.Report(ctx, cos_pa, rca_pa)
    post, heads = fit_heads(ctx, partition_labels(ctx.groups, scorer_train), scorer_train,
                            SMOKE_HEAD_ROWS if smoke else HEAD_ROWS)
    np.savez_compressed(res / "n6_posteriors.npz", selection=ctx.selection,
                        **{f"{h}__{m}": post[h][m][ctx.selection] for h in PARTS for m in ("img", "txt")})
    hard, soft, uni, info, soft_info = n6_terms(post, ep)
    arrays = {"anchor_group": cl, "pair_index": ctx.pair_index}
    out = {"heads": heads, "term_only": {}}
    for name, s in (("T6", hard), ("T6soft", soft), ("T6u", uni)):
        rc.assert_finite_scores(s, name)
        pa = per_anchor(s)
        for m in METRICS:
            arrays[f"term_{name}__{m}"] = np.asarray(pa[m])
        out["term_only"][name] = {**rep.describe(pa), "flat_share": rc.flat_share(s)}
    nested, control, picks = crossfit_nested(ctx.cos, uni, hard, ctx.parity)
    pn, pc = per_anchor(nested), per_anchor(control)
    for kind, pa in (("nested", pn), ("control", pc)):
        for m in METRICS:
            arrays[f"{kind}__{m}"] = np.asarray(pa[m])
    out["nested"] = {**rep.full(pn, control=pc), "picks": picks, "per_pair": ctx.per_pair(pn)}
    out["control"] = rep.describe(pc)
    vc = out["nested"]["vs_control"]
    out["passes"] = config_passes(vc["r1"], vc["gain"])
    out["picked_partition"] = {
        rg.POOLED_ORDER[i]: {c: {h: 100 * float(np.mean(info[c]["weights"][ctx.pair_index == i].argmax(axis=1) == j))
                                 for j, h in enumerate(PARTS)} for c in ("a", "b")} for i in range(len(rg.PAIRS))}
    out["soft_fallback_share"] = 100 * float(np.mean([soft_info[c]["fallback"] for c in ("a", "b")]))
    ad = res / "per_anchor_addendum1.npz"
    if ad.exists():                                     # descriptive: the strongest condition-free score (ADDENDUM_1)
        z = np.load(ad)
        pm = {m: z[f"dev__A3__matched__{m}"].astype(np.float64) for m in METRICS}
        out["reference_A3_matched_control"] = rep.describe(pm)
        out["nested"]["vs_A3_matched_control"] = rep.paired(pn, pm)
    return out, arrays


def test_stage(smoke, res):
    dev = json.loads((res / "n6_seed42.json").read_text())
    if not dev["passes"] and not smoke:
        raise SystemExit("N6 did not pass the seed-42 check (ADDENDUM_2_N6.md §4): no test")
    seeds = SMOKE_TEST_SEEDS if smoke else TEST_SEEDS
    ad = res / "per_anchor_addendum1.npz"
    ref = np.load(ad) if ad.exists() else None
    pooled = {s: {m: [] for m in METRICS} for s in (*SCORERS, "A3_matched")}
    clusters, per_seed, picks_all, arrays = [], {}, {}, {}
    for seed in seeds:
        ctx = rg.EvalContext(seed, smoke)
        cl = ctx.anchor_group
        post = load_posteriors(res / "n6_posteriors.npz", ctx)
        cos_pa, rca_pa = e1_arrays(ctx, smoke, seed)
        hard, _, uni, _, _ = n6_terms(post, ctx.pooled)
        for name, s in (("T6", hard), ("T6u", uni)):
            rc.assert_finite_scores(s, f"seed {seed} {name}")
        nested, control, picks = crossfit_nested(ctx.cos, uni, hard, ctx.parity)
        pa = {"config": per_anchor(nested), "control": per_anchor(control), "cosine": cos_pa, "rca": rca_pa}
        key = f"test__seed{seed}__matched__r1"
        has_ref = ref is not None and key in ref.files
        if has_ref:
            pa["A3_matched"] = {m: ref[f"test__seed{seed}__matched__{m}"].astype(np.float64) for m in METRICS}
        block = {s: {"summary": summarize(pa[s], cl), "either": rc.point_ci(pa[s]["r1"] + pa[s]["other"], cl)}
                 for s in SCORERS}
        block["vs"] = {c: {m: compare(pa["config"], pa[c], cl, m) for m in ("r1", "gain")}
                       for c in ("cosine", "rca", "control")}
        if has_ref:
            block["vs_A3_matched_control"] = {m: compare(pa["config"], pa["A3_matched"], cl, m) for m in ("r1", "gain")}
        per_seed[str(seed)] = block
        picks_all[str(seed)] = picks
        for s in pa:
            for m in METRICS:
                arrays[f"seed{seed}__{s}__{m}"] = np.asarray(pa[s][m])
                pooled[s][m].append(np.asarray(pa[s][m], dtype=np.float64))
        clusters.append(cl)
        rc.log(f"test seed {seed} done")
    cl = np.concatenate(clusters)
    P = {s: {m: np.concatenate(v) for m, v in d.items()} for s, d in pooled.items() if d["r1"]}
    pooled_block = {s: {"summary": summarize(P[s], cl), "either": rc.point_ci(P[s]["r1"] + P[s]["other"], cl)}
                    for s in SCORERS}
    pooled_block["vs"] = {c: {m: compare(P["config"], P[c], cl, m) for m in ("r1", "gain")}
                          for c in ("cosine", "rca", "control")}
    if "A3_matched" in P and len(P["A3_matched"]["r1"]) == len(cl):
        pooled_block["vs_A3_matched_control"] = {m: compare(P["config"], P["A3_matched"], cl, m) for m in ("r1", "gain")}
    pooled_block["n_episodes"], pooled_block["n_clusters"] = int(len(cl)), int(len(np.unique(cl)))
    return {"seeds": list(seeds), "per_seed": per_seed, "picks": picks_all, "pooled": pooled_block,
            "go": go_verdict(pooled_block["vs"]), "dev_passed": dev["passes"]}, arrays


def dev_text(r):
    def c(x):
        return f"{x['point']:6.2f} [{x['ci95'][0]:6.2f}, {x['ci95'][1]:6.2f}]"

    lines = [f"N6 seed-42 check{' (SMOKE)' if r['smoke'] else ''}",
             "heads (held-out accuracy img / txt): " + "; ".join(
                 f"{h} {e['heldout_accuracy']['img']:.1f} / {e['heldout_accuracy']['txt']:.1f}"
                 for h, e in r["heads"].items() if h in PARTS)]
    for name, x in r["term_only"].items():
        lines.append(f"  term {name:7s} R@1 {c(x['summary']['r1'])} gain {c(x['summary']['gain'])} "
                     f"either {x['either']['point']:6.2f} flat {x['flat_share']:5.2f}")
    n = r["nested"]
    lines.append(f"  control R@1 {c(r['control']['summary']['r1'])} either {r['control']['either']['point']:6.2f}")
    lines.append(f"  nested  R@1 {c(n['summary']['r1'])} gain {c(n['summary']['gain'])} either {n['either']['point']:6.2f} "
                 f"picks {n['picks']}")
    for comp in ("control", "cosine", "rca"):
        lines.append(f"  nested - {comp:7s} R@1 {c(n['vs_' + comp]['r1'])} gain {c(n['vs_' + comp]['gain'])}")
    if "vs_A3_matched_control" in n:
        lines.append(f"  nested - A3 matched control (descriptive) R@1 {c(n['vs_A3_matched_control']['r1'])}")
    lines.append(f"  picked partition per pair/condition: {r['picked_partition']}")
    lines.append(f"PASS (vs own control): {r['passes']}")
    return "\n".join(lines)


def test_text(r):
    def c(x):
        return f"{x['point']:6.2f} [{x['ci95'][0]:6.2f}, {x['ci95'][1]:6.2f}]"

    lines = [f"N6 fresh-seed test, seeds {r['seeds']}{' (SMOKE)' if r['smoke'] else ''}"]
    for key in [*(str(s) for s in r["seeds"]), "pooled"]:
        b = r["per_seed"][key] if key != "pooled" else r["pooled"]
        lines.append(f"{key}:")
        for s in SCORERS:
            lines.append(f"  {s:7s} R@1 {c(b[s]['summary']['r1'])} gain {c(b[s]['summary']['gain'])} "
                         f"either {b[s]['either']['point']:6.2f}")
        for comp in ("cosine", "rca", "control"):
            lines.append(f"  config - {comp:7s} R@1 {c(b['vs'][comp]['r1'])} gain {c(b['vs'][comp]['gain'])}")
        if "vs_A3_matched_control" in b:
            lines.append(f"  config - A3 matched control (descriptive) R@1 {c(b['vs_A3_matched_control']['r1'])}")
    lines.append(f"GO: {r['go']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("dev", "test"), required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    res = HERE / "results" / ("smoke" if args.smoke else "")
    res.mkdir(parents=True, exist_ok=True)
    stem = "n6_seed42" if args.stage == "dev" else "n6_test"
    outs = [res / f"{stem}.json", res / f"{stem}.txt", res / f"per_anchor_{stem}.npz"]
    if not args.smoke and any(p.exists() for p in outs):
        raise SystemExit(f"{stem} results exist in {res}; refusing to overwrite")
    t0 = time.time()
    result, arrays = (dev_stage if args.stage == "dev" else test_stage)(args.smoke, res)
    result["smoke"] = args.smoke
    result["provenance"] = {"addendum_sha256": rg.sha_file(HERE / "ADDENDUM_2_N6.md"),
                            "script_sha256": rg.sha_file(Path(__file__)),
                            "module_sha256": rg.sha_file(rc.ROOT / "src/eval/aspect_quick_checks.py"),
                            "partitions_sha256": PARTITIONS_SHA}
    rg.assert_finite_tree(result)
    outs[0].write_text(json.dumps(result, indent=1))
    np.savez_compressed(outs[2], **arrays)
    text = (dev_text if args.stage == "dev" else test_text)(result)
    outs[1].write_text(text + "\n")
    print(text)
    rc.log(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
