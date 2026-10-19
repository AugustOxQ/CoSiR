"""Report bitwise Stage 1 repeat agreement without gating on differences."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(first, second, out):
    out = Path(out)
    out.unlink(missing_ok=True)
    first, second = Path(first), Path(second)
    name = 'stage1/clip_roberta__s0.npz'
    first_npz, second_npz = first / name, second / name
    with np.load(first_npz, allow_pickle=False) as left, np.load(second_npz, allow_pickle=False) as right:
        arrays = {}
        for key in sorted(set(left.files) | set(right.files)):
            if key not in left.files or key not in right.files:
                arrays[key] = False
                continue
            a, b = left[key], right[key]
            arrays[key] = (a.dtype == b.dtype and a.shape == b.shape
                           and a.tobytes() == b.tobytes())
    left = json.loads(first_npz.with_suffix('.json').read_text())
    right = json.loads(second_npz.with_suffix('.json').read_text())
    sidecar = {key: (key in left and key in right
                     and json.dumps(left[key], sort_keys=True)
                     == json.dumps(right[key], sort_keys=True))
               for key in sorted((left.keys() | right.keys()) - {'seconds'})}
    result = dict(bitwise_equal=all(arrays.values()) and all(sidecar.values()),
                  arrays=arrays, sidecar=sidecar, first=str(first), second=str(second),
                  first_npz_sha256=hashlib.sha256(first_npz.read_bytes()).hexdigest(),
                  second_npz_sha256=hashlib.sha256(second_npz.read_bytes()).hexdigest())
    pending = out.with_suffix('.pending.json')
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    pending.replace(out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    compare(args.first, args.second, args.out)


if __name__ == '__main__':
    main()
