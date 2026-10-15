"""Reference arrays from round 1's common.load_bundle() on seed 42 (round 4 §5 item 1 and §8: the re-derivation may
import it, seed 42 only). Its console output (it prints step-1 values) is discarded. Saves only the arrays that item 1
compares against: episodes, parity, anchor paintings, pair index, cosine, B (scores and per-anchor), B'(A0) and
B'(A1) (scores and per-anchor), the posteriors of affect, image, caption and csd on selection rows, and the SHA-256 of
the full csd posterior arrays (NaN outside selection rows). Separate process from rd4_bundle.py, so that the two module
import graphs never mix. Writes out/rd4_ref42.npz and .json. Usage: python rd4_ref42.py
"""
import contextlib
import io
import json
import time

import numpy as np

import rd4_core as K
from rd4_core import COND, DIRS, METRICS, T

NEEDED = ["20261117_reader_fix_csd/common.py", "20261117_reader_fix_csd/DECISION_RULE.md",
          "20261030_aspect_baselines/results/episodes_seed42.npz",
          "20261111_community_told_oracle/results/per_anchor_told_oracle.npz",
          "20261031_pseudo_partitions/results/partitions.npz",
          "20261116_grouping_step1_style/results/step1_group_style.npz",
          "20261108_new_method_quick_checks/results/n6_posteriors.npz",
          "20261116_grouping_step1_style/results/step1_heads_style.npz",
          "20261116_grouping_step1_style/results/step1_eval_style.npz",
          "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt"]
OUTNPZ = K.OUT / "rd4_ref42.npz"
OUTJSON = K.OUT / "rd4_ref42.json"


def main():
    t0 = time.time()
    K.guard_seed(42)
    rec = {"rules": K.assert_rules(), "inputs_sha256": K.assert_inputs(NEEDED)}
    common = K.import_path("common", T / "20261117_reader_fix_csd/common.py")
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        bd = common.load_bundle(smoke=False)
    rec["load_bundle_stdout_lines_discarded"] = len(sink.getvalue().splitlines())
    del sink
    ctx = bd.ctx
    ep = ctx.pooled
    sel = np.asarray(ctx.selection)
    npz = {"anchor": ep.anchor, "candidates": ep.candidates, "parity": np.asarray(ctx.parity),
           "anchor_group": np.asarray(bd.cl), "pair_index": np.asarray(ctx.pair_index), "selection": sel}
    for c in COND:
        for d in DIRS:
            npz[f"cos__{c}__{d}"] = np.asarray(ctx.cos[c][d])
            npz[f"B__{c}__{d}"] = np.asarray(bd.B[c][d])
            npz[f"Bp0__{c}__{d}"] = np.asarray(bd.Bp["A0"][c][d])
            npz[f"Bp1__{c}__{d}"] = np.asarray(bd.Bp["A1"][c][d])
    for m in METRICS:
        npz[f"pB__{m}"] = np.asarray(bd.pB[m])
        npz[f"pBp0__{m}"] = np.asarray(bd.pBp["A0"][m])
        npz[f"pBp1__{m}"] = np.asarray(bd.pBp["A1"][m])
    for h in K.A1:
        for m in ("img", "txt"):
            npz[f"post__{h}__{m}"] = np.asarray(bd.post[h][m])[sel]
    rec["csd_full_sha256"] = {m: K.sha_arr(np.asarray(bd.post["csd"][m])) for m in ("img", "txt")}
    rec["csd_shape_dtype"] = {m: [list(np.asarray(bd.post["csd"][m]).shape), str(np.asarray(bd.post["csd"][m]).dtype)]
                              for m in ("img", "txt")}
    rec["load_bundle_checks"] = K.jsonable(bd.checks)
    rec["affect_head"] = K.jsonable(bd.affect_head)
    rec["runtime_s"] = round(time.time() - t0, 1)
    rec["written_amsterdam"] = K.now_ams()
    np.savez(OUTNPZ, **npz)
    rec["cache_sha256"] = K.sha_file(OUTNPZ)
    K.save_json(OUTJSON, rec)
    K.log(f"round 1's load_bundle references written ({rec['runtime_s']} s)", t0)


if __name__ == "__main__":
    main()
