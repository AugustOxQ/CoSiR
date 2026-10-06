"""Seed-42 cache for the round-2 re-derivation (rule §4.1).

Loads round 1's bundle (common.load_bundle: standard heads, B, B'(A0, A1), step-1 regression check, ~72 s), then with
our own code: the 18 (A0) and 24 (A1) seed-42 reader features of both conditions, and the grouping scores s_h of
affect, image, caption, csd per direction. Checks: our features equal rb_eval.seed42_features exactly; our s_h equal
common.grouping_stack exactly; B is condition-free; Delta^b = -Delta^a. Writes out/rd2_seed42_cache.npz and
out/rd2_prep.json.
"""
import time

import numpy as np

import rd2_core as K
from rd2_core import COND, DIRS, METRICS, RC


def main():
    t0 = time.time()
    RC.assert_rule()
    K.check_base_constants()
    bundle = RC.C.load_bundle(smoke=False)
    K.log("bundle loaded", t0)
    ctx = bundle.ctx
    ep = ctx.pooled
    post = bundle.post
    rec = {"bundle_checks": bundle.checks, "n_episodes": int(len(ctx.parity)),
           "n_clusters": int(len(np.unique(ctx.anchor_group)))}
    npz = {"parity": np.asarray(ctx.parity), "anchor_group": np.asarray(ctx.anchor_group),
           "pair_index": np.asarray(ctx.pair_index)}
    # B and comparators
    for c in COND:
        for d in DIRS:
            npz[f"B__{c}__{d}"] = np.asarray(bundle.B[c][d])
    rec["B_dtype"] = str(npz["B__a__i2t"].dtype)
    rec["B_condition_free"] = all(np.array_equal(bundle.B["a"][d], bundle.B["b"][d]) for d in DIRS)
    if not rec["B_condition_free"]:
        raise SystemExit("B differs between conditions")
    for m in METRICS:
        npz[f"pB__{m}"] = np.asarray(bundle.pB[m], np.float64)
        for cfg in K.CONFIGS:
            npz[f"pBp_{cfg}__{m}"] = np.asarray(bundle.pBp[cfg][m], np.float64)
    rec["r1_means"] = {"B": 100 * float(np.mean(npz["pB__r1"])),
                       **{f"Bprime_{cfg}": 100 * float(np.mean(npz[f"pBp_{cfg}__r1"])) for cfg in K.CONFIGS}}
    rec["r1_means_rule"] = {"B": 18.341064453125, "Bprime_A0": 18.436686197916664, "Bprime_A1": 18.804931640625}
    rec["r1_means_equal_rule"] = {k: rec["r1_means"][k] == v for k, v in rec["r1_means_rule"].items()}
    # own features vs rb_eval.seed42_features
    rec["features"] = {}
    for cfg, parts in K.CONFIGS.items():
        F = K.features_both(post, parts, ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt)
        Fr, chk = RC.rbe.seed42_features(bundle, parts)
        eq = {c: bool(np.array_equal(F[c], Fr[c])) for c in COND}
        dneg = all(np.array_equal(F["b"][:, 6 * j + 2], -F["a"][:, 6 * j + 2]) for j in range(len(parts)))
        rec["features"][cfg] = {"own_equal_rb_eval_seed42_features": eq, "rb_eval_checks": chk,
                                "delta_b_equals_minus_delta_a": bool(dneg), "shape": list(F["a"].shape),
                                "max_abs_diff": float(max(np.abs(F[c] - Fr[c]).max() for c in COND))}
        if not all(eq.values()) or not dneg:
            raise SystemExit(f"{cfg}: own seed-42 features differ from rb_eval.seed42_features: {rec['features'][cfg]}")
        for c in COND:
            npz[f"F_{cfg}__{c}"] = F[c]
        K.log(f"{cfg} features equal rb_eval.seed42_features", t0)
    # own grouping scores s_h = p_h(query) . p_h(candidate), float32 einsum, query's own modality
    G = K.CONFIGS["A1"]
    own = {"i2t": np.stack([np.einsum("nc,nkc->nk", post[h]["img"][ep.anchor], post[h]["txt"][ep.candidates])
                            for h in G], axis=1),
           "t2i": np.stack([np.einsum("nc,nkc->nk", post[h]["txt"][ep.anchor], post[h]["img"][ep.candidates])
                            for h in G], axis=1)}
    ref = RC.C.grouping_stack(post, ep, G)
    rec["stack_equal_common_grouping_stack"] = {d: bool(np.array_equal(own[d], ref[d])) for d in DIRS}
    rec["stack_dtype"] = str(own["i2t"].dtype)
    if not all(rec["stack_equal_common_grouping_stack"].values()):
        raise SystemExit("own grouping scores differ from common.grouping_stack")
    for d in DIRS:
        npz[f"stack__{d}"] = own[d]
    npz["stack_groupings"] = np.array(G)
    np.savez(K.CACHE, **npz)
    rec["cache_sha256"] = RC.sha_file(K.CACHE)
    rec["runtime_s"] = round(time.time() - t0, 1)
    rec["provenance"] = K.provenance()
    K.save_json(K.OUT / "rd2_prep.json", rec)
    K.log(f"cache written: {rec['r1_means']} equal rule {rec['r1_means_equal_rule']}", t0)


if __name__ == "__main__":
    main()
