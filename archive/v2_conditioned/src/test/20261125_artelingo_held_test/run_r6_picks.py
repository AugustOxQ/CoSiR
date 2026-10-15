"""Stage 1b, first half, of round 6 (DECISION_RULE.md section 6 item 3, section 5 item 4, section 11; contracts
section 5; ticket 06): on seed 42 (selection rows, the refit heads' posteriors) cross-fit B, B'(A0) and B'(A1) with
crossfit_condition_free, keep its picks per tune half, and write them with the mean R@1 to results/picks_seed42.json.

Checks, each recorded per comparator or scorer (any failure is a difference):
  mean_r1_equal_target    the mean R@1 equals 18.341064453125 (B), 18.436686197916664 (B0), 18.804931640625 (B1)
                          exactly (rule section 6 item 3)
  frozen_equals_crossfit  r6_score's frozen assembly with these picks reproduces the cross-fit bit for bit
  lambda_convention       crossfit_lambda rerun on seed 42 for RCA and each PM scorer gives baselines_seed42.json's
                          lambda_picks key for key ("0" = tuned on half 0, scores parity 1)

Selection rows only: no held row is loaded beyond the split's index arrays. CPU only, 8 threads (asserted):

    cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/run_r6_picks.py \
        > src/test/20261125_artelingo_held_test/results/run_r6_picks.log 2>&1

Prints pass or FAIL per check and the file path, never a mean R@1. Exit 0: every check passed and
results/picks_seed42.json is written. Exit 3: a difference; the record goes to results/picks_seed42_failed.json
(picks_seed42.json is not written; nothing is fixed; the difference is traced and the user decides).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402
import r6_context as X  # noqa: E402
import r6_heads as H  # noqa: E402
import r6_picks as P  # noqa: E402

EXIT_DIFF = 3
OUT = R.RESULTS / P.PICKS_NAME


def build_bundle_seed42():
    """The seed-42 selection bundle on the round-6 path: refit heads, RowContext, fit_pm, build_bundle_r6."""
    H.require_threads()
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import artelingo_aspect_labels
    data = load_artelingo()
    split = R.load_split(data)
    labels = artelingo_aspect_labels(data)
    vs = R.development_value_sets(labels, split.groups, split.selection)
    heads = H.fit_heads_r6(data, split.groups, split.scorer_train)
    ctx = X.RowContext("selection", R.DEV_SEED, data, split, labels, heads, vs, R.N_PER_PAIR)
    pm = B.fit_pm(data, split.scorer_train)
    return B.build_bundle_r6(ctx, B.load_readers(), pm)


def failed_path(out) -> Path:
    out = Path(out)
    return out.with_name(f"{out.stem}_failed{out.suffix}")


def run(bundle, out) -> int:
    """Cross-fit, check, write ``out`` (or its _failed twin) and print pass or FAIL per check; -> exit code."""
    out = Path(out)
    picks, scores = P.crossfit_seed42(bundle)
    lams = P.frozen_lambdas()
    checks = {"frozen_equals_crossfit": P.frozen_equals_crossfit(bundle, picks, scores),
              "lambda_convention": P.lambda_convention(bundle, lams)}
    rec = P.picks_record(picks, checks)
    rec["lambda_picks"] = {name: {str(h): ("inf" if math.isinf(v) else v) for h, v in lam.items()}
                           for name, lam in lams.items()}
    for k in ("episodes_sha256", "coef_sha256", "fit_rows_sha256", "input_sha256"):
        rec[k] = getattr(bundle, k, None)
    code = 0
    if not rec["passed"]:
        code = EXIT_DIFF  # guard:exit_code
    path = out if code == 0 else failed_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    for chk, items in rec["checks"].items():
        for name, ok in items.items():
            print(f"{chk} {name}: {'pass' if ok else 'FAIL'}")
    print(f"picks seed 42: {'PASSED' if rec['passed'] else 'FAILED'}; record {path}")
    if code:
        print(f"stage-1b stop (rule section 6 item 3): exit {code}")
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="output JSON (default results/picks_seed42.json)")
    args = ap.parse_args(argv)
    t0 = time.time()
    code = run(build_bundle_seed42(), args.out)
    print(f"runtime {time.time() - t0:.0f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
