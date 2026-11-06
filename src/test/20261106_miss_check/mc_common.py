"""Read-only r23 inputs and r24 output/interval helpers."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROUND = Path(__file__).resolve().parent
sys.path.insert(0, str(ROUND.parent / '20261106_text_claim'))
import tcl_paths as P
from tcl_metrics import painting_interval

R23 = P.DEFAULT_OUT
OUT = Path(os.environ.get('MC_OUTPUTS', '/data/SSD2/cosir_r24/20261106_miss_check'))
assert not OUT.resolve().is_relative_to(R23.resolve())
assert not OUT.resolve().is_relative_to(P.R15_OUT.resolve())


def record(group):
    return P.read_committed_record(f'stage1_{group}.json')


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def interval(c, m, pid):
    point, lower, upper, n = painting_interval(c, m, pid, draws=10000)
    return dict(point=point, lower=lower, upper=upper, n_units=n, n_query=len(pid))


def matched(c, m):
    mask = np.isfinite(m)
    assert mask.any() and not np.isinf(m).any()
    assert all(np.array_equal(np.isfinite(x), mask) and not np.isinf(x).any() for x in c)
    return mask


def saved(set_name, kind, row, head, seed, direction, readout, aspect):
    path = P.score_path(set_name, kind, row, head, seed, direction, readout, out=R23)
    with np.load(path, allow_pickle=False) as z:
        return z['q_pid'], z[f'ap__{aspect}'].astype(np.float64)
