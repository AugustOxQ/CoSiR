"""Seed-42 bundle for the round-4 re-derivation (round 4 §4 item 1; round 3 §4 item 1), from the loaders and frozen
components that the rules' §8 lets the re-derivation import, in round 3's call sequence, then the A1 extension:

  ctx = EvalContext(42, False); scorer_train = artelingo_splits(ctx.data).scorer_train
  T_N1u = centered_term(run_checks.model_inputs(ctx, "A3", scorer_train, False)[0], ctx.pooled, uniform=True)
  post_E2 = run_n6.load_posteriors(n6_posteriors.npz, ctx)
  B  = crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post_E2, ep, E2's three groupings), parity)[0]
  affect heads = run_told_oracle.fit_one_head(ctx, global_labels(partition_L, ...), scorer_train, run_n6.HEAD_ROWS)
  B'(A0) = crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post_A0, ep, A0), parity)[0]
  A1 extension (round 4 D2, D4): z = load(step1_heads_style.npz), z["selection"] == ctx.selection asserted,
  post["csd"] = {img: run_step1.full_post(z, "style_csd__img", ctx), txt: ... "style_csd__txt"}, finite on selection
  rows; B'(A1) = crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post, ep, A1), parity)[0]
  then our own grouping scores s_h (A0 and A1), our own 18 A0 and 24 A1 features (rd3_core.features_both), v (D5).

Seed 42 only. Writes out/rd4_bundle_seed42.npz and .json. Usage: python rd4_bundle.py
"""
import json
import time

import numpy as np

import rd4_core as K
from rd4_core import COND, DIRS, K3, T

rg = K.import_path("run_gonogo", T / "20261101_aspect_factor_gonogo/run_gonogo.py")
rc = K.import_path("run_checks", T / "20261108_new_method_quick_checks/run_checks.py")
n6 = K.import_path("run_n6", T / "20261108_new_method_quick_checks/run_n6.py")
rto = K.import_path("run_told_oracle", T / "20261111_community_told_oracle/run_told_oracle.py")
rbb = K.import_path("rb_build", T / "20261117_reader_fix_csd/rb_build.py")
rs1 = K.import_path("run_step1", T / "20261116_grouping_step1_style/run_step1.py")

from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_quick_checks import centered_term, crossfit_condition_free, uniform_probe_scores  # noqa: E402

N6_POST = T / "20261108_new_method_quick_checks/results/n6_posteriors.npz"
TOLD_NPZ = T / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz"
TOLD_JSON = T / "20261111_community_told_oracle/results/told_oracle.json"
HEADS_STYLE = T / "20261116_grouping_step1_style/results/step1_heads_style.npz"
E2_PARTS = ("affect", "image", "caption")
NEEDED = ["20261030_aspect_baselines/results/episodes_seed42.npz",
          "20261111_community_told_oracle/results/per_anchor_told_oracle.npz",
          "20261031_pseudo_partitions/results/partitions.npz",
          "20261108_new_method_quick_checks/results/n6_posteriors.npz",
          "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt",
          "20261111_community_told_oracle/results/told_oracle.json",
          "20261117_reader_fix_csd/rb_build.py",
          "20261117_reader_fix_csd/common.py",
          "20261117_reader_fix_csd/results/rb_reader_A0.pkl",
          "20261117_reader_fix_csd/results/rb_reader_A0.json",
          "20261117_reader_fix_csd/results/rb_reader_A1.pkl",
          "20261117_reader_fix_csd/results/rb_reader_A1.json",
          "20261117_reader_fix_csd/results/rb_reader_A1.npz",
          "20261116_grouping_step1_style/run_step1.py",
          "20261116_grouping_step1_style/results/step1_heads_style.npz"]
CACHE = K.OUT / "rd4_bundle_seed42.npz"
RECORD = K.OUT / "rd4_bundle_seed42.json"


def check_modules():
    """The imported modules are the rule's (not another run_checks / common on sys.path)."""
    import sys
    want = {"run_gonogo": T / "20261101_aspect_factor_gonogo/run_gonogo.py",
            "run_checks": T / "20261108_new_method_quick_checks/run_checks.py",
            "run_n6": T / "20261108_new_method_quick_checks/run_n6.py",
            "run_told_oracle": T / "20261111_community_told_oracle/run_told_oracle.py",
            "rb_build": T / "20261117_reader_fix_csd/rb_build.py",
            "common": T / "20261117_reader_fix_csd/common.py",
            "run_step1": T / "20261116_grouping_step1_style/run_step1.py"}
    got = {k: str(K.Path(sys.modules[k].__file__).resolve()) for k in want}
    for k, p in want.items():
        if got[k] != str(p.resolve()):
            raise AssertionError(f"module {k} loaded from {got[k]}, not {p}")
    if tuple(n6.PARTS) != E2_PARTS:
        raise AssertionError(f"run_n6.PARTS {n6.PARTS} differs from E2's three groupings")
    if rs1.full_post.__module__ != "run_step1":
        raise AssertionError("full_post is not run_step1's")
    return got


def build(seed=42):
    K.guard_seed(seed)
    t0 = time.time()
    rec = {"seed": int(seed), "rules": K.assert_rules(), "inputs_sha256": K.assert_inputs(NEEDED),
           "modules": check_modules()}
    ctx = rg.EvalContext(seed, False)
    ep = ctx.pooled
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    K.log(f"seed {seed}: EvalContext ({ctx.n} episodes), scorer_train {len(scorer_train)}", t0)
    T_N1u = centered_term(rc.model_inputs(ctx, "A3", scorer_train, False)[0], ep, uniform=True)
    post_E2 = n6.load_posteriors(N6_POST, ctx)
    B, B_picks = crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post_E2, ep, E2_PARTS), ctx.parity)
    K.log("T_N1u, B", t0)
    zt = np.load(TOLD_NPZ)
    lab = rto.global_labels(zt["partition_L"], scorer_train, len(ctx.groups))
    aff, aff_prov = rto.fit_one_head(ctx, lab, scorer_train, n6.HEAD_ROWS)
    stored_L = json.loads(TOLD_JSON.read_text())["arms"]["L"]["head"]
    rt = json.loads(json.dumps(K.jsonable(aff_prov)))
    rec["affect_head_equals_told_oracle_L"] = bool(rt == stored_L)
    if not rec["affect_head_equals_told_oracle_L"]:
        raise SystemExit(f"affect head {rt} differs from told_oracle.json arm L's head {stored_L}")
    K.log("affect heads (identity with told_oracle.json arm L asserted)", t0)
    post = {"affect": aff, "image": post_E2["image"], "caption": post_E2["caption"]}
    Bp0, Bp0_picks = crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post, ep, K.A0), ctx.parity)
    # ---- A1 extension (round 4 D2, D4)
    zh = np.load(HEADS_STYLE)
    rec["heads_style_selection_equal_ctx"] = bool(np.array_equal(zh["selection"], ctx.selection))
    if not rec["heads_style_selection_equal_ctx"]:
        raise AssertionError("step-1 heads were computed on another selection")
    post["csd"] = {"img": rs1.full_post(zh, "style_csd__img", ctx), "txt": rs1.full_post(zh, "style_csd__txt", ctx)}
    for g in K.A1:
        for m in ("img", "txt"):
            if not np.isfinite(post[g][m][ctx.selection]).all():
                raise AssertionError(f"{g}/{m}: non-finite posterior on selection rows")
    rec["csd_full_sha256"] = {m: K.sha_arr(post["csd"][m]) for m in ("img", "txt")}
    rec["csd_shape_dtype"] = {m: [list(post["csd"][m].shape), str(post["csd"][m].dtype)] for m in ("img", "txt")}
    rec["csd_nonfinite_rows_outside_selection"] = {
        m: int((~np.isfinite(post["csd"][m]).all(axis=1)).sum()) for m in ("img", "txt")}
    rec["n_rows_outside_selection"] = int(len(ctx.groups) - int(np.asarray(ctx.selection).sum())
                                          if np.asarray(ctx.selection).dtype == bool
                                          else len(ctx.groups) - len(ctx.selection))
    Bp1, Bp1_picks = crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post, ep, K.A1), ctx.parity)
    K.log("B'(A0), csd posteriors, B'(A1)", t0)
    stack0 = K3.grouping_scores(post, ep, K.A0)
    stack1 = K3.grouping_scores(post, ep, K.A1)
    F0 = K3.features_both(post, K.A0, ep)
    F1 = K3.features_both(post, K.A1, ep)
    for c in COND:
        if not np.array_equal(F1[c][:, :18], F0[c]):
            raise AssertionError("first 18 A1 features differ from the A0 features")
    for F, H in ((F0, 3), (F1, 4)):
        if not all(np.array_equal(F["b"][:, 6 * j + 2], -F["a"][:, 6 * j + 2]) for j in range(H)):
            raise AssertionError("Delta^b != -Delta^a")
    for name, s in (("B", B), ("Bprime_A0", Bp0), ("Bprime_A1", Bp1)):
        if not all(np.array_equal(s["a"][d], s["b"][d]) for d in DIRS):
            raise AssertionError(f"{name} is not condition-free")
    v, vb = K.abstention_signal(F0["a"], F0["b"])
    rec["v_b_equals_v_a"] = bool(np.array_equal(v, vb))
    if not rec["v_b_equals_v_a"]:
        raise AssertionError("min(S_image^b, C_image^b) != min(S_image^a, C_image^a)")
    rec["picks"] = {"B": B_picks, "Bprime_A0": Bp0_picks, "Bprime_A1": Bp1_picks}
    sel = np.asarray(ctx.selection)
    npz = {"anchor": ep.anchor, "candidates": ep.candidates, "parity": np.asarray(ctx.parity),
           "anchor_group": np.asarray(ctx.anchor_group), "pair_index": np.asarray(ctx.pair_index),
           "selection": sel, "v": v}
    for c in COND:
        for d in DIRS:
            npz[f"cos__{c}__{d}"] = np.asarray(ctx.cos[c][d])
            npz[f"B__{c}__{d}"] = np.asarray(B[c][d])
            npz[f"Bp0__{c}__{d}"] = np.asarray(Bp0[c][d])
            npz[f"Bp1__{c}__{d}"] = np.asarray(Bp1[c][d])
        npz[f"F0__{c}"] = F0[c]
        npz[f"F1__{c}"] = F1[c]
    for d in DIRS:
        npz[f"stack0__{d}"] = stack0[d]
        npz[f"stack1__{d}"] = stack1[d]
    for h in K.A1:
        for m in ("img", "txt"):
            npz[f"post__{h}__{m}"] = post[h][m][sel]
    rec["dtypes"] = {"B": str(npz["B__a__i2t"].dtype), "Bp1": str(npz["Bp1__a__i2t"].dtype),
                     "stack": str(stack0["i2t"].dtype), "cos": str(npz["cos__a__i2t"].dtype),
                     "F": str(F0["a"].dtype)}
    rec["n_episodes"], rec["n_clusters"] = int(ctx.n), int(len(np.unique(ctx.anchor_group)))
    rec["episodes_sha256"] = ctx.shas
    rec["runtime_s"] = round(time.time() - t0, 1)
    rec["written_amsterdam"] = K.now_ams()
    np.savez(CACHE, **npz)
    rec["cache_sha256"] = K.sha_file(CACHE)
    K.save_json(RECORD, rec)
    K.log(f"bundle seed {seed} written ({rec['runtime_s']} s)", t0)
    return rec


def load():
    rec = json.loads(RECORD.read_text())
    if K.sha_file(CACHE) != rec["cache_sha256"]:
        raise SystemExit("bundle cache changed since it was written")
    z = np.load(CACHE)
    sc = lambda p: {c: {d: z[f"{p}__{c}__{d}"] for d in DIRS} for c in COND}  # noqa: E731
    b = {"parity": z["parity"], "cl": z["anchor_group"], "pair_index": z["pair_index"], "anchor": z["anchor"],
         "candidates": z["candidates"], "selection": z["selection"], "v": z["v"],
         "cos": sc("cos"), "B": sc("B"), "Bp0": sc("Bp0"), "Bp1": sc("Bp1"),
         "F0": {c: z[f"F0__{c}"] for c in COND}, "F1": {c: z[f"F1__{c}"] for c in COND},
         "stack0": {d: z[f"stack0__{d}"] for d in DIRS}, "stack1": {d: z[f"stack1__{d}"] for d in DIRS},
         "post": {h: {m: z[f"post__{h}__{m}"] for m in ("img", "txt")} for h in K.A1}}
    return b, rec


def main():
    K.assert_rules()
    build(42)


if __name__ == "__main__":
    main()
