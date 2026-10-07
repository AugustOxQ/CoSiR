"""Round 5 placement step (rule DECISION_RULE.md section 5 item 3 and D4), once, CPU.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python run_r5_placement.py [--item3-only]

Item 3: the placement function equals run_told_oracle.fit_one_head's txt and img posteriors exactly, its head record
equals told_oracle.json arm L's head, and scorer_train[pos] == draw. --item3-only stops there (record in
results/smoke/item3_check.json, no GE head). Otherwise the GE head (D4, with its fallback), Q_GE and
results/placement.json follow. The console prints pass or fail and SHA-256s only; the GE head's numbers go to
results/placement.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r5_common as R5  # noqa: E402
import r5_placement as P  # noqa: E402

TOLD_JSON = "20261111_community_told_oracle/results/told_oracle.json"
AFFECT_NPZ = "src/test/20261018_affect_factor_learning/cache/affect_prepare.npz"
GOEMO_NPZ = R5.CACHE / "r5_goemotions_selection.npz"
GE_NPZ = R5.CACHE / "r5_ge_posterior.npz"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def item3(ctx, lab, scorer_train):
    RTO, N6, RCHK, RG = R5.RTO, R5.N6, R5.RCHK, R5.RG
    post_ref, prov = RTO.fit_one_head(ctx, lab, scorer_train, N6.HEAD_ROWS)
    R5.R3.assert_inputs([TOLD_JSON])
    stored = json.loads(R5.R3.input_path(TOLD_JSON).read_text())["arms"]["L"]["head"]
    feats = {"txt": ctx.data.txt_features, "img": ctx.data.img_features}
    rec, ok = {"n_iter": {}, "heldout_accuracy": {}, "equals_fit_one_head": {}, "classes_ok": {}}, True
    draws = {}
    for m, F in feats.items():
        res = P.place(F, RCHK.unit, lab, scorer_train, ctx.selection, R5.GE_MAX_ITER)
        same = (res["post"].shape == (R5.N_ROWS, R5.N_CLASSES) and res["post"].dtype == np.float32
                and np.array_equal(res["post"], post_ref[m], equal_nan=True))
        acc_ok = res["acc"] == prov["heldout_accuracy"][m] \
            and round(res["acc"], 2) == R5.CLIP_HEAD_RECORD["heldout_accuracy"][m]
        rec["equals_fit_one_head"][m] = bool(same)
        rec["heldout_accuracy"][m] = res["acc"]
        rec["n_iter"][m] = res["n_iter"]
        rec["classes_ok"][m] = bool(np.array_equal(res["classes"], np.arange(R5.N_CLASSES)))
        rec[f"accuracy_equals_record_{m}"] = bool(acc_ok)
        draws[m] = res["draw"]
        ok = ok and same and acc_ok and rec["classes_ok"][m]
        log(f"item 3 {m}: {'equal' if same and acc_ok else 'DIFFERENT'}")
    pos = P.draw_positions(scorer_train, draws["txt"])
    rec["draw_positions_ok"] = bool(np.array_equal(scorer_train[pos], draws["txt"])
                                    and np.array_equal(draws["txt"], draws["img"]))
    rec["head_roundtrip_equals_told_oracle_arm_L"] = bool(RTO.roundtrip(prov) == stored)
    rec["head_equals_constants"] = bool(
        prov["n_classes"] == R5.CLIP_HEAD_RECORD["n_classes"]
        and prov["draw_rows_sha256"] == R5.CLIP_HEAD_RECORD["draw_rows_sha256"]
        and abs(prov["check_majority_share"] - R5.CLIP_HEAD_RECORD["check_majority_share"]) < 5e-3
        and prov["uniform"] == R5.CLIP_HEAD_RECORD["uniform"])
    rec["clip_n_iter_expected_descriptive"] = R5.CLIP_N_ITER
    ok = ok and rec["draw_positions_ok"] and rec["head_roundtrip_equals_told_oracle_arm_L"] and rec["head_equals_constants"]
    rec["passed"] = bool(ok)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item3-only", action="store_true")
    args = ap.parse_args()
    R5.assert_rule()
    R5.assert_modules()
    R5.assert_inputs([AFFECT_NPZ])
    out_json = R5.SMOKE / "item3_check.json" if args.item3_only else R5.RESULTS / "placement.json"
    if not args.item3_only:
        R5.refuse_existing([out_json, GE_NPZ], False)
        if R5.GOEMO_FILE_SHA is None:
            raise SystemExit("r5_common.GOEMO_FILE_SHA is not set: no GE head is fitted")
    from src.data.artelingo_splits import artelingo_splits

    ctx = R5.RG.EvalContext(42, False)          # data and row sets only; no episode field is read
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    with np.load(R5.C.INPUT_FILES["affect_L_partition(per_anchor_told_oracle)"]) as z:
        partition_L = np.asarray(z["partition_L"], dtype=np.int64)
    lab = R5.RTO.global_labels(partition_L, scorer_train, R5.N_ROWS)
    if len(ctx.groups) != R5.N_ROWS:
        raise SystemExit("unexpected row count")
    rec3 = item3(ctx, lab, scorer_train)
    print(f"item 3: {'PASS' if rec3['passed'] else 'FAIL'}", flush=True)
    rec = {"rule_sha256": R5.RULE_SHA, "time": R5.now_ams(), "item3": rec3}
    if args.item3_only or not rec3["passed"]:
        R5.write_json_once(out_json, rec, smoke=args.item3_only)
        print(f"wrote {out_json.name} sha256 {R5.sha256_file(out_json)}")
        sys.exit(0 if rec3["passed"] else 1)
    with np.load(R5.input_path(AFFECT_NPZ)) as z:
        affect_probs = np.asarray(z["affect_probs"])
    X = P.ge_input(ctx, GOEMO_NPZ, scorer_train, affect_probs)
    res, head = P.fit_ge_head(X, lab, scorer_train, ctx.selection)
    post_sel = res["post"][ctx.selection]
    sha = P.write_ge_posterior(GE_NPZ, post_sel, np.asarray(ctx.selection, dtype=np.int64), res["classes"])
    rec["ge_head"] = head
    rec["ge_posterior_sha256"] = sha
    R5.write_json_once(out_json, rec, smoke=False)
    print(f"GE head fitted; r5_ge_posterior.npz sha256 {sha}; placement.json sha256 {R5.sha256_file(out_json)}")


if __name__ == "__main__":
    main()
