"""AFF's full CLIP recipe chain with the image and caption k-means groupings replaced by Leiden groupings.

  lg_rebuild.py --step {groups,a3,reader,baselines,cells,score,all} [--reb DIR]

Runs the frozen worktree's recipe code (/project/CoSiR-lg-r9, commit b84db1e) in this process, never writing under it.
The swap: r9_recipe_groups.fit_partitions returns lg_group.npz's image_leiden64 / caption_leiden64 (affect and
local_groups stay CLIP's). The redirect: BB.CLIP_PARTS recipe roles point into DIR (default results/rebuild), and
R.input_pins maps every recipe-output input to its file in DIR (live SHA-256) once that file exists; frozen inputs keep
their R6 pins. The content graph is CLIP's own (it depends on features and scorer_train only), reloaded, not rebuilt.
Skipped by design: step-1 loss / pinned-input / pinned-checkpoint gates of A3, CLIP-pinned picks, lambdas and mean-R@1
targets of the baselines, rule-TAUS and pinned-cells checks of the cells step.  Kept: all other checks.
Only the groups step needs no GPU (graph reused); a3 needs the GPU lock.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WT = "/project/CoSiR-lg-r9/src/test/20261202_r9_scope"
sys.path.insert(0, WT)
HERE = Path(__file__).resolve().parent
LABELS = HERE / "results/lg_group.npz"

import numpy as np  # noqa: E402
import r9_common as R  # noqa: E402
import r9_backbone as BB  # noqa: E402

assert str(Path(R.HERE).resolve()) == WT

B = "clip_b32"
ORIG = dict(BB.CLIP_PARTS)
MAP = {}
REB = None


def install(reb):
    global REB
    REB = Path(reb).resolve()
    REB.mkdir(parents=True, exist_ok=True)
    assert not REB.is_relative_to(Path(WT)) and not REB.is_relative_to(R.TEST / "20261202_r9_scope")
    for role in BB.RECIPE_ROLES:
        p = ORIG[role]
        name = "cells_seed42.json" if role == "cells_seed42" else p.name
        BB.CLIP_PARTS[role] = REB / name
        if role not in ("baselines_seed42", "cells_seed42") and Path(p).is_relative_to(R.TEST):
            MAP[str(Path(p).relative_to(R.TEST))] = REB / name
    base = R.input_pins

    def input_pins(backbone="clip_b32"):
        out = dict(base(backbone))
        if backbone == "clip_b32":
            for rel, p in MAP.items():
                if rel in out and p.is_file():
                    out[rel] = (p, R._sha_cached(p))
        return out
    R.input_pins = input_pins


def patch_picks():
    import r9_picks as P

    def load_picks(path=None, backbone="clip_b32"):
        path = Path(path or BB.parts(backbone)["picks_seed42"])
        rec = json.loads(path.read_text())
        assert rec.get("passed") is True and all(type(rec["mean_r1"].get(n)) is float for n in P.NESTED), path
        assert rec.get("backbone") == B and rec.get("targets") is None, f"{path}: not a rebuild picks record"
        return {n: P.as_nested_pick(rec[n], f"{path.name} {n}") for n in P.NESTED}
    P.load_picks = load_picks


# ---------------------------------------------------------------- steps

def step_groups(stop_after="heads"):
    import r9_recipe_groups as G
    import r9_heads as H
    from scipy.sparse import load_npz

    def load_frozen():
        p = ORIG["affect_km"]
        assert R.sha256_file(p) == R.INPUT_SHA256[H.E2_PARTITIONS_REL], "CLIP partitions.npz differs from its pin"
        with np.load(p, allow_pickle=False) as z:
            return {k: np.array(z[k]) for k in ("affect", "local_groups")}

    def fit_partitions(img, txt, groups, scorer_train, frozen, **kw):
        st = np.asarray(scorer_train, dtype=np.int64)
        with np.load(LABELS) as z:
            assert np.array_equal(z["scorer_train"], st), "lg_group.npz scorer_train != the chain's"
            lab = {k: np.asarray(z[k], dtype=np.int64) for k in ("image_leiden64", "caption_leiden64")}
        local_groups = np.asarray(frozen["local_groups"], dtype=np.int64)
        affect = np.asarray(frozen["affect"], dtype=np.int64)
        assert np.array_equal(local_groups, np.unique(np.asarray(groups)[st], return_inverse=True)[1])
        out = {"affect": affect, "local_groups": local_groups}
        for name, key in (("image", "image_leiden64"), ("caption", "caption_leiden64")):
            x = lab[key]
            assert x.shape == (len(st),) and x.dtype == np.int64 and x.min() == 0, name
            c = np.bincount(x)
            assert len(c) == 64 and c.min() >= 200, f"{name}: {len(c)} groups, min {c.min()}"
            out[name] = x
        return {"affect": out["affect"], "image": out["image"], "caption": out["caption"],
                "local_groups": out["local_groups"]}

    def fit_graph(img, txt, scorer_train, allow_cpu=False):
        p = ORIG["graph"]
        assert R.sha256_file(p) == json.loads(R.pins_path(B).read_text())["parts"]["graph"]["sha256"], "CLIP graph differs from its pin"
        g = load_npz(p).tocsr()
        assert g.shape == (len(scorer_train),) * 2
        return g

    G.load_frozen, G.fit_partitions, G.fit_graph = load_frozen, fit_partitions, fit_graph
    G.graph_device_record = lambda allow_cpu=False: {"device": "none", "reused": "CLIP graph.npz (feature-only input)"}
    return G.main(["--backbone", B, "--out", str(REB), "--stop-after", stop_after])


def step_a3():
    import r9_recipe_a3 as A

    def inputs_gate(backbone, loaded):
        return {"passed": True, "pinned": False, "sha256": loaded["sha"],
                "features": {"img_scorer_train": A.R.rg.sha_array(loaded["img"]),
                             "txt_scorer_train": A.R.rg.sha_array(loaded["txt"])},
                "feature_store_files": A.store_shas(backbone)}
    A.inputs_gate = inputs_gate
    A.step1_gate = lambda history: {"passed": None, "note": "skipped: pinned to the k-means inputs"}
    return A.main(["--backbone", B, "--run", "main", "--out", str(REB), "--inputs", str(REB)])


def step_reader():
    import r9_recipe_reader as RD
    return RD.main(["--backbone", B, "--out", str(REB), "--stage", "all", "--upstream", str(REB)])


def step_baselines():
    import r9_recipe_baselines as BL
    import r9_picks as P
    patch_picks()

    def finish(bundle, out, backbone, input_sha256):
        out = Path(out)
        BL.preflight(out, backbone)
        P.check_seed42(bundle)
        lams = BL.crossfit_lambdas(bundle)
        base = BL.baselines_record(bundle, lams)
        parsed = P.parse_lambda_picks(json.loads(json.dumps(base)))
        picks, scores = P.crossfit_seed42(bundle)
        checks = {"frozen_equals_crossfit": P.frozen_equals_crossfit(bundle, picks, scores),
                  "lambda_convention": P.lambda_convention(bundle, parsed)}
        rec = P.picks_record(picks, checks)
        del rec["checks"]["mean_r1_equal_target"]
        rec["targets"] = None
        rec["passed"] = all(bool(v) for chk in rec["checks"].values() for v in chk.values())
        rec.update({"backbone": backbone, "lambda_picks": base["lambda_picks"],
                    "episodes_sha256": dict(bundle.episodes_sha256), "coef_sha256": bundle.coef_sha256,
                    "fit_rows_sha256": bundle.fit_rows_sha256, "input_sha256": dict(input_sha256)})
        out.mkdir(parents=True, exist_ok=True)
        for chk, items in rec["checks"].items():
            for name, ok in items.items():
                print(f"{chk} {name}: {'pass' if ok else 'FAIL'}", flush=True)
        if not rec["passed"]:
            BL._write_json(out / BL.FAILED, rec)
            return BL.EXIT_DIFF
        picks_made = ["B", "B0", "B1", "mean_r1", "checks", "passed", "rule_sha256", "module_sha256", "time", "backbone",
                      "lambda_picks", "coef_sha256", "input_sha256"]
        picks_copied = [k for k in rec if k not in picks_made]
        BL._write_with_record(out, BL.FILES[0], base, backbone, input_sha256, BL.BASELINES_MADE, BL.BASELINES_COPIED)
        BL._write_with_record(out, BL.FILES[1], rec, backbone, input_sha256, picks_made, picks_copied)
        print(f"picks seed 42: PASSED; written {out / BL.FILES[0]} and {out / BL.FILES[1]}")
        return 0
    BL.finish = finish
    return BL.main(["--backbone", B, "--out", str(REB), "--partitions-dir", str(REB), "--a3-dir", str(REB)])


def step_cells():
    import time
    import r9_recipe_cells as CE
    patch_picks()

    def load_taus(backbone):
        taus = json.loads(Path(BB.parts(backbone)["rc_tau"]).read_text())["taus"]
        assert len(taus) == CE.N_TAU and all(type(t) is float for t in taus)
        return taus
    CE.load_taus = load_taus
    out = REB
    CE.preflight(out)
    t0 = time.time()
    taus = CE.load_taus(B)
    env = CE.setup(B)
    assert env.head_check["passed"], "refit heads differ from the rebuilt posteriors"
    ctx = CE.X.RowContext("selection", R.DEV_SEED, env.data, env.split, env.labels, env.heads, env.value_sets,
                          R.N_PER_PAIR)
    CE.check_episode_sha(ctx.episode_sha)
    bundle = CE.cells_bundle(ctx, B)
    rec, details = CE.compute_cells(bundle, env.readers, taus, B)
    from r9_recipe_groups import read_shas
    path = CE.write_cells(out, rec, backbone=B, input_sha256=env.inputs,
                          upstream_sha256={r: R.sha256_file(BB.parts(B)[r]) for r in CE.UPSTREAM_ROLES},
                          extra={"feature_sha256": read_shas(B), "taus": taus, "details": details,
                                 "episodes_sha256": dict(ctx.episode_sha), "time": R.amsterdam_now(),
                                 "runtime_s": round(time.time() - t0, 1)})
    print(f"written {path}")
    print(json.dumps(rec))
    return 0


def step_score():
    import r9_bundle as BU
    import r9_context as X
    import r9_heads as H
    import r9_picks as P
    import r9_score as S
    import run_r9_stage0a as S0A
    patch_picks()
    paths = BB.parts(B)
    H.require_threads()
    env = R.load_rows(B)
    heads = H.fit_heads_r9(env.data, env.split.groups, env.split.scorer_train, backbone=B)
    chk = H.check_selection(heads, env.data, env.split.selection, B)
    assert chk["passed"], "head selection check failed against the rebuilt posteriors"
    pm = BU.fit_pm(env.data, env.split.scorer_train, B)
    readers = BU.load_readers(B)
    R.assert_inputs(B)
    ctx = X.RowContext("selection", 42, env.data, env.split, env.labels, heads, env.value_sets, R.N_PER_PAIR)
    bundle = BU.build_bundle_r6(ctx, readers, pm, B)
    P.check_seed42(bundle)
    read = lambda k: json.loads(Path(paths[k]).read_text())  # noqa: E731
    scored = S.score_seed(bundle, P.load_picks(paths["picks_seed42"], B), P.parse_lambda_picks(read("baselines_seed42")),
                          readers, include_pm=True, cells=S.cells_from_record(read("cells_seed42")),
                          taus=tuple(read("rc_tau")["taus"]))
    cl, pi = np.asarray(bundle.cl), np.asarray(bundle.pair_index)
    nums, bar_v = S0A.number_items(scored, cl, pi)
    C = S0A.C
    f64 = lambda x: np.asarray(x, np.float64)  # noqa: E731
    aff, b0 = f64(scored["aff_fused"]["r1"]), f64(scored["B0"]["r1"])
    arrays = {"cl": cl, "pair_index": pi, "aff_fused_r1": aff, "aff_cf_r1": f64(scored["aff_cf"]["r1"]),
              "B0_r1": b0, "B_r1": f64(scored["B"]["r1"]), "B1_r1": f64(scored["B1"]["r1"]), "bar_v_aff": f64(bar_v["aff"])}
    np.savez(REB / "lg_seed42_arrays.npz", **arrays)
    result_from_arrays()
    return 0


def result_from_arrays():
    """lg_result.json from lg_seed42_arrays.npz and round 3's k-means arrays (no rescoring)."""
    import run_r9_stage0a as S0A
    C = S0A.C
    f64 = lambda x: np.asarray(x, np.float64)  # noqa: E731
    ci = lambda x, cl: (lambda d: [float(d["point"]), *map(float, d["ci95"])])(C.point_ci(x, cl))  # noqa: E731
    with np.load(REB / "lg_seed42_arrays.npz") as z:
        a = {k: f64(z[k]) if k not in ("cl", "pair_index") else np.asarray(z[k]) for k in z.files}
    cl, pi, aff, b0 = a["cl"], a["pair_index"], a["aff_fused_r1"], a["B0_r1"]
    means = {k: 100 * float(a[k].mean()) for k in ("aff_fused_r1", "aff_cf_r1", "B0_r1", "B_r1", "B1_r1")}
    res = {"leiden": {**means, "margin_vs_Bprime": ci(aff - b0, cl), "bar_margin": ci(a["bar_v_aff"], cl),
                      "margin_vs_matched_control": ci(aff - a["aff_cf_r1"], cl)}}
    ref = R.INPUT_PATHS[S0A.R3_ARRAYS_REL]
    with np.load(ref) as z:
        k_aff, k_bar = f64(z["aff_fused__r1"]), f64(z["aff_bar_v"])
        assert np.array_equal(z["cl"], cl) and np.array_equal(z["pair_index"], pi), "episodes differ from the k-means run"
    res["kmeans"] = {"aff_fused_r1": 100 * float(k_aff.mean()), "bar_margin": ci(k_bar, cl)}
    res["paired_leiden_minus_kmeans"] = {"aff_fused_r1": ci(aff - k_aff, cl), "bar_margin": ci(a["bar_v_aff"] - k_bar, cl)}
    (REB / "lg_result.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


def summary():
    r = json.loads((REB / "lg_result.json").read_text())
    f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"  # noqa: E731
    L, K, D = r["leiden"], r["kmeans"], r["paired_leiden_minus_kmeans"]
    print(f"AFF seed-42 dev, R@1 points: bar margin Leiden {f(L['bar_margin'])} | k-means {f(K['bar_margin'])} | paired "
          f"{f(D['bar_margin'])}; AFF R@1 Leiden {L['aff_fused_r1']:.2f}, k-means {K['aff_fused_r1']:.2f}, paired "
          f"{f(D['aff_fused_r1'])}; Leiden over B'(A0) {f(L['margin_vs_Bprime'])}, over its matched control "
          f"{f(L['margin_vs_matched_control'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("groups", "a3", "reader", "baselines", "cells", "score", "result", "summary"), required=True)
    ap.add_argument("--reb", default=str(HERE / "results/rebuild"))
    ap.add_argument("--stop-after", default="heads", help="groups step only (check runs)")
    a = ap.parse_args()
    assert os.environ.get("OMP_NUM_THREADS") == "8", "set OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"
    install(a.reb)
    if a.step == "groups":
        return step_groups(a.stop_after)
    if a.step == "summary":
        summary()
        return 0
    return {"a3": step_a3, "reader": step_reader, "baselines": step_baselines, "cells": step_cells,
            "score": step_score, "result": lambda: result_from_arrays() or 0}[a.step]()


if __name__ == "__main__":
    code = main()
    if code == 0 and "--step" in sys.argv and sys.argv[sys.argv.index("--step") + 1] in ("score", "result"):
        summary()
    sys.exit(code)
