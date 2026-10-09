"""Stage 1a of round 6 (DECISION_RULE.md §6 item 2, §5 item 5, §11; contracts §3; ticket 03): refit the five posterior
heads on the same 60,000 scorer-train rows, compare their selection-row posteriors bit for bit with the stored arrays
(n6_posteriors.npz's six, step1_heads_style.npz's `style_csd__img` and `style_csd__txt`), check the affect heads' prov
against told_oracle.json arm L (R3 rule D2), and record each head's coefficient SHA-256 for the read (rule §5.5).

Selection rows only: no held row is loaded or predicted here. CPU only, 8 threads (asserted):

    cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/run_r6_refit.py \
        > src/test/20261125_artelingo_held_test/results/run_r6_refit.log 2>&1

Writes results/refit_check.json (contracts §3). Prints pass or FAIL per item and the file paths, never a posterior or
an accuracy. Exit 0: every item passed. Exit 3: a difference, a stage-1a stop (rule §6.2: nothing is fixed; the
differences, count and max abs diff per key, are in the JSON and in results/refit_check_differences.txt; the user
decides).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_heads as H  # noqa: E402

EXIT_DIFF = 3
OUT = R.RESULTS / "refit_check.json"
DIFF_NAME = "refit_check_differences.txt"


def threads_record() -> dict:
    """Thread settings of this process: the environment and the BLAS / OpenMP pools threadpoolctl sees."""
    from threadpoolctl import threadpool_info
    return {"env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
            "pools": sorted({(p.get("internal_api"), int(p.get("num_threads"))) for p in threadpool_info()})}


def record(chk, heads, inputs, extra=None) -> dict:
    """refit_check.json's content (contracts §3), plus the heads' prov, convergence-warning counts and threads."""
    passed = bool(all(it["equal"] for it in chk["items"].values()) and chk["affect_identity"])
    H._require(passed == chk["passed"], "check_selection's passed flag disagrees with its items")
    rec = {"passed": passed, "items": chk["items"], "affect_identity": bool(chk["affect_identity"]),
           "coef_sha256": H.coef_sha256(heads), "module_sha256": R.r6_module_shas(), "input_sha256": inputs,
           "time": R.amsterdam_now(),
           "prov": {h: heads[h]["prov"] for h in heads},
           "convergence_warnings": {h: heads[h]["convergence_warnings"] for h in heads}}
    rec.update(extra or {})
    return rec


def report(rec, out) -> int:
    """Write ``rec`` to ``out``; print pass or FAIL per item and the paths; on any difference write the differences
    (count and max abs diff per key) next to it and return EXIT_DIFF, else 0."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    for key, it in rec["items"].items():
        print(f"{key}: {'pass' if it['equal'] else 'FAIL'}")
    print(f"affect_identity: {'pass' if rec['affect_identity'] else 'FAIL'}")
    print(f"refit check: {'PASSED' if rec['passed'] else 'FAILED'}; record {out}")
    code = 0
    if not rec["passed"]:
        code = EXIT_DIFF  # guard:exit_code
    if code:
        diff = out.parent / DIFF_NAME
        lines = [f"{key}: n_diff {it['n_diff']}, max_abs_diff {it['max_abs_diff']!r}"
                 + (f", shape {it['shape']}, dtype {it['dtype']}" if "shape" in it else "")
                 for key, it in rec["items"].items() if not it["equal"]]
        if not rec["affect_identity"]:
            lines.append(f"affect_identity: prov differs from told_oracle.json arm L: {rec['prov']['affect']!r}")
        diff.write_text("\n".join(lines) + "\n")
        print(f"stage-1a stop (rule §6.2): differences in {diff}; exit {code}")
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="output JSON (default results/refit_check.json)")
    args = ap.parse_args(argv)
    H.require_threads()
    t0 = time.time()
    inputs = R.assert_inputs()
    from src.data.artelingo import load_artelingo
    data = load_artelingo()
    split = R.load_split(data)
    heads = H.fit_heads_r6(data, split.groups, split.scorer_train)
    chk = H.check_selection(heads, data, split.selection)
    rec = record(chk, heads, inputs, {"threads": threads_record(), "runtime_s": round(time.time() - t0, 1),
                                      "n_selection": int(len(split.selection))})
    return report(rec, args.out)


if __name__ == "__main__":
    sys.exit(main())
