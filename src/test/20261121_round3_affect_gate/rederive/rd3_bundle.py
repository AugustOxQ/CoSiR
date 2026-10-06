"""Seed-parameterised bundle for the round-3 re-derivation (rule §4 item 1), built only from the loaders and frozen
components that rule §8 lets the re-derivation import, in the rule's call sequence:

  ctx = EvalContext(s, False); scorer_train = artelingo_splits(ctx.data).scorer_train
  T_N1u = centered_term(run_checks.model_inputs(ctx, "A3", scorer_train, False)[0], ctx.pooled, uniform=True)
  post_E2 = run_n6.load_posteriors(n6_posteriors.npz, ctx)
  B  = crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post_E2, ep, E2's three groupings), parity)[0]
       (run_n6.n6_terms(...)[2] is exactly uniform_probe_scores(post, ep, run_n6.PARTS); asserted PARTS equal)
  affect heads = run_told_oracle.fit_one_head(ctx, global_labels(partition_L, scorer_train, n), scorer_train,
       run_n6.HEAD_ROWS), its record asserted equal to told_oracle.json arm L's head
  B'(A0) = crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post_A0, ep, A0), parity)[0]
  then our own grouping scores s_h and our own 18 reader features.

Never calls round 1's common.load_bundle or run_sweep.setup. Writes out/rd3_bundle_seed{s}.npz and .json.
Usage: python rd3_bundle.py --seed 42
"""
import argparse
import json
import time

import numpy as np

import rd3_core as K
from rd3_core import COND, DIRS, T

rg = K.import_path("run_gonogo", T / "20261101_aspect_factor_gonogo/run_gonogo.py")
rc = K.import_path("run_checks", T / "20261108_new_method_quick_checks/run_checks.py")
n6 = K.import_path("run_n6", T / "20261108_new_method_quick_checks/run_n6.py")
rto = K.import_path("run_told_oracle", T / "20261111_community_told_oracle/run_told_oracle.py")
rbb = K.import_path("rb_build", T / "20261117_reader_fix_csd/rb_build.py")

from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_quick_checks import centered_term, crossfit_condition_free, uniform_probe_scores  # noqa: E402

N6_POST = T / "20261108_new_method_quick_checks/results/n6_posteriors.npz"
TOLD_NPZ = T / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz"
TOLD_JSON = T / "20261111_community_told_oracle/results/told_oracle.json"
NEEDED = ["20261030_aspect_baselines/results/episodes_seed42.npz",
          "20261111_community_told_oracle/results/per_anchor_told_oracle.npz",
          "20261031_pseudo_partitions/results/partitions.npz",
          "20261108_new_method_quick_checks/results/n6_posteriors.npz",
          "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt",
          "20261111_community_told_oracle/results/told_oracle.json",
          "20261117_reader_fix_csd/rb_build.py",
          "20261117_reader_fix_csd/results/rb_reader_A0.pkl",
          "20261117_reader_fix_csd/results/rb_reader_A0.json"]


def cache_path(seed):
    return K.OUT / f"rd3_bundle_seed{seed}.npz"


def check_modules():
    """The imported modules are the rule's (not another run_checks / common on sys.path)."""
    want = {"run_gonogo": T / "20261101_aspect_factor_gonogo/run_gonogo.py",
            "run_checks": T / "20261108_new_method_quick_checks/run_checks.py",
            "run_n6": T / "20261108_new_method_quick_checks/run_n6.py",
            "run_told_oracle": T / "20261111_community_told_oracle/run_told_oracle.py",
            "rb_build": T / "20261117_reader_fix_csd/rb_build.py",
            "common": T / "20261117_reader_fix_csd/common.py"}
    import sys
    got = {k: str(K.Path(sys.modules[k].__file__).resolve()) for k in want}
    for k, p in want.items():
        if got[k] != str(p.resolve()):
            raise AssertionError(f"module {k} loaded from {got[k]}, not {p}")
    if tuple(n6.PARTS) != K.E2_PARTS:
        raise AssertionError(f"run_n6.PARTS {n6.PARTS} differs from E2's three groupings")
    return got


def build(seed):
    K.guard_seed(seed)
    t0 = time.time()
    rec = {"seed": int(seed), "inputs_sha256": K.assert_inputs(NEEDED), "modules": check_modules()}
    ctx = rg.EvalContext(seed, False)
    ep = ctx.pooled
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    K.log(f"seed {seed}: EvalContext ({ctx.n} episodes), scorer_train {len(scorer_train)}", t0)
    T_N1u = centered_term(rc.model_inputs(ctx, "A3", scorer_train, False)[0], ep, uniform=True)
    K.log("T_N1u", t0)
    post_E2 = n6.load_posteriors(N6_POST, ctx)
    T6u = uniform_probe_scores(post_E2, ep, K.E2_PARTS)
    B, B_picks = crossfit_condition_free(ctx.cos, T_N1u, T6u, ctx.parity)
    K.log(f"B (picks {B_picks})", t0)
    zt = np.load(TOLD_NPZ)
    lab = rto.global_labels(zt["partition_L"], scorer_train, len(ctx.groups))
    aff, aff_prov = rto.fit_one_head(ctx, lab, scorer_train, n6.HEAD_ROWS)
    stored_L = json.loads(TOLD_JSON.read_text())["arms"]["L"]["head"]
    rt = json.loads(json.dumps(aff_prov))
    rec["affect_head"] = rt
    rec["affect_head_equals_told_oracle_L"] = bool(rt == stored_L)
    if not rec["affect_head_equals_told_oracle_L"]:
        raise SystemExit(f"affect head {rt} differs from told_oracle.json arm L's head {stored_L}")
    K.log("affect heads (identity with told_oracle.json arm L asserted)", t0)
    post = {"affect": aff, "image": post_E2["image"], "caption": post_E2["caption"]}
    Bp, Bp_picks = crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post, ep, K.A0), ctx.parity)
    K.log(f"B'(A0) (picks {Bp_picks})", t0)
    stack = K.grouping_scores(post, ep, K.A0)
    F = K.features_both(post, K.A0, ep)
    dneg = all(np.array_equal(F["b"][:, 6 * j + 2], -F["a"][:, 6 * j + 2]) for j in range(3))
    if not dneg:
        raise AssertionError("Delta^b != -Delta^a")
    for name, s in (("B", B), ("Bprime", Bp)):
        if not all(np.array_equal(s["a"][d], s["b"][d]) for d in DIRS):
            raise AssertionError(f"{name} is not condition-free")
    pk, prec, _ = rbb.load_readers("A0", False)
    rec["reader_C"] = [float(h["model"].C) for h in pk["halves"]]
    rec["picks"] = {"B": B_picks, "Bprime": Bp_picks}
    npz = {"anchor": ep.anchor, "candidates": ep.candidates, "parity": np.asarray(ctx.parity),
           "anchor_group": np.asarray(ctx.anchor_group), "pair_index": np.asarray(ctx.pair_index),
           "selection": ctx.selection}
    for c in COND:
        for d in DIRS:
            npz[f"cos__{c}__{d}"] = np.asarray(ctx.cos[c][d])
            npz[f"B__{c}__{d}"] = np.asarray(B[c][d])
            npz[f"Bp__{c}__{d}"] = np.asarray(Bp[c][d])
        npz[f"F__{c}"] = F[c]
    for d in DIRS:
        npz[f"stack__{d}"] = stack[d]
    for h in K.A0:
        for m in ("img", "txt"):
            npz[f"post__{h}__{m}"] = post[h][m][ctx.selection]
    rec["dtypes"] = {"B": str(npz["B__a__i2t"].dtype), "stack": str(stack["i2t"].dtype), "cos": str(npz["cos__a__i2t"].dtype)}
    rec["n_episodes"], rec["n_clusters"] = int(ctx.n), int(len(np.unique(ctx.anchor_group)))
    rec["episodes_sha256"] = ctx.shas
    rec["runtime_s"] = round(time.time() - t0, 1)
    rec["written_amsterdam"] = K.now_ams()
    np.savez(cache_path(seed), **npz)
    rec["cache_sha256"] = K.sha_file(cache_path(seed))
    K.save_json(K.OUT / f"rd3_bundle_seed{seed}.json", rec)
    K.log(f"bundle seed {seed} written", t0)
    return rec


def load(seed):
    K.guard_seed(seed)
    z = np.load(cache_path(seed))
    b = {"parity": z["parity"], "cl": z["anchor_group"], "pair_index": z["pair_index"], "anchor": z["anchor"],
         "candidates": z["candidates"], "selection": z["selection"],
         "cos": {c: {d: z[f"cos__{c}__{d}"] for d in DIRS} for c in COND},
         "B": {c: {d: z[f"B__{c}__{d}"] for d in DIRS} for c in COND},
         "Bp": {c: {d: z[f"Bp__{c}__{d}"] for d in DIRS} for c in COND},
         "F": {c: z[f"F__{c}"] for c in COND}, "stack": {d: z[f"stack__{d}"] for d in DIRS},
         "post": {h: {m: z[f"post__{h}__{m}"] for m in ("img", "txt")} for h in K.A0}}
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    K.assert_rule()
    build(a.seed)


if __name__ == "__main__":
    main()
