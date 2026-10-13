"""CPU head replay; compare every output array with the backbone's stored parts."""

import argparse
from pathlib import Path
import tempfile
import time

import r9_common as R
import r9_heads as H
from r9_backbone import BACKBONES, parts

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    args = parser.parse_args(argv)
    H.require_threads()
    start = time.monotonic()
    rows = R.load_rows(args.backbone)
    heads = H.fit_heads_r9(rows.data, rows.split.groups, rows.split.scorer_train, backbone=args.backbone)
    checked = H.check_selection(heads, rows.data, rows.split.selection, args.backbone)
    passed, count = checked["passed"], 0
    with tempfile.TemporaryDirectory(prefix="heads-", dir=R.HERE / "results") as directory:
        H.write_posteriors(directory, heads, rows.data, rows.split.selection, backbone=args.backbone,
                           script_path=__file__)
        for role in ("n6_posteriors", "step1_heads_style"):
            path = parts(args.backbone)[role]
            R.assert_input(H.N6_REL if role == "n6_posteriors" else H.CSD_HEADS_REL, args.backbone)
            with np.load(Path(directory) / path.name, allow_pickle=False) as got, np.load(path, allow_pickle=False) as want:
                equal_keys = set(got.files) == set(want.files)
                passed &= equal_keys
                print(f"{role}/keys: {'pass' if equal_keys else 'FAIL'}", flush=True)
                for key in want.files:
                    equal = (key in got.files and got[key].dtype == want[key].dtype
                             and got[key].shape == want[key].shape and np.array_equal(got[key], want[key]))
                    passed &= equal
                    count += 1
                    print(f"{role}/{key}: {'pass' if equal else 'FAIL'}", flush=True)
    print(f"affect_identity: {'pass' if checked['affect_identity'] else 'FAIL'}" if args.backbone == "clip_b32"
          else "affect_identity: CLIP only", flush=True)
    print(f"{args.backbone}: {count} arrays, {'PASS' if passed else 'FAIL'} ({time.monotonic() - start:.1f}s)", flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    R.RESULTS.mkdir(parents=True, exist_ok=True)
    raise SystemExit(main())
