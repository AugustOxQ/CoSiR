"""A1 ablation (DECISION_RULE.md section 4.8; descriptive, decides nothing): the A1 - A0 difference of the same reader,
paired per anchor, for the fused R@1 and the bar margin, with the painting-clustered interval.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261118_reader_fix_round2/r2_a1_ablation.py --reader R1

Reads results/cand_<R>_{A0,A1}.{json,npz} (SHA-256 of each npz checked against its JSON) and results/rule_application.json
(the ablation runs only after the rule has been applied, on the reader it names). Writes results/a1_ablation_<R>.json.
"""
import argparse
import json

import numpy as np

import r2_common as R


def load(name):
    rec = json.loads((R.RES / f"cand_{name}.json").read_text())
    if rec["provenance"]["rule_sha256"] != R.RULE_SHA:
        raise SystemExit(f"cand_{name}.json was written under another rule")
    npz = R.RES / f"cand_{name}.npz"
    if R.sha_file(npz) != rec["npz_sha256"]:
        raise SystemExit(f"{npz} differs from its JSON (SHA-256)")
    return rec, np.load(npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, choices=("R1", "R2", "R3"))
    args = ap.parse_args()
    R.assert_rule()
    app = json.loads((R.RES / "rule_application.json").read_text())
    if app["rule_sha256"] != R.RULE_SHA:
        raise SystemExit("rule_application.json was written under another rule")
    if app["ablation_candidate"] != args.reader:
        raise SystemExit(f"the rule application names {app['ablation_candidate']} for the ablation, not {args.reader}")
    out = R.RES / f"a1_ablation_{args.reader}.json"
    R.refuse_existing([out], smoke=False)
    (r0, z0), (r1, z1) = load(f"{args.reader}_A0"), load(f"{args.reader}_A1")
    cl = R.C.load_bundle(smoke=False).cl
    if len(cl) != len(z0["fused__r1"]) or len(cl) != len(z1["fused__r1"]):
        raise AssertionError("episode counts differ")
    diff = {
        "fused_r1": R.C.point_ci(z1["fused__r1"].astype(np.float64) - z0["fused__r1"].astype(np.float64), cl),
        "bar_margin": R.C.point_ci(z1["bar_v"].astype(np.float64) - z0["bar_v"].astype(np.float64), cl),
    }
    rec = {"reader": args.reader, "role": app["ablation_label"],
           "note": "A1 minus A0, paired per anchor, pp; descriptive (rule section 4.8), decides nothing",
           "a1_minus_a0": diff,
           "inputs": {f"cand_{args.reader}_{c}.npz": R.sha_file(R.RES / f"cand_{args.reader}_{c}.npz") for c in ("A0", "A1")}}
    R.write_json_once(out, rec, smoke=False)
    print(json.dumps(diff, indent=1))


if __name__ == "__main__":
    main()
