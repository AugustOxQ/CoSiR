"""Unit tests for the pure helpers of validate_v1_stage2.py. Synthetic arrays
only -- no real data, GPU, pilot modules or W&B are touched.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_v1_stage2 import classify_label_diffs, compare_targets, v1_verdict


def test_compare_targets_exact_and_mismatch():
    a = np.eye(3, dtype=np.float32)
    assert compare_targets(a, a.copy()) == {"equal": True, "n_diff_rows": 0}
    b = a.copy(); b[1] = [0, 0, 1]
    assert compare_targets(a, b) == {"equal": False, "n_diff_rows": 1}


def test_compare_targets_returns_plain_python_types_and_counts_rows_not_cells():
    a = np.zeros((4, 5), dtype=np.float32)
    b = a.copy()
    b[0, :3] = 1.0       # three cells, one row
    b[3, 4] = 1.0
    out = compare_targets(a, b)
    assert out == {"equal": False, "n_diff_rows": 2}
    assert type(out["equal"]) is bool and type(out["n_diff_rows"]) is int


def test_compare_targets_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compare_targets(np.eye(3), np.eye(4))


def test_v1_verdict_pass_needs_equal_targets_and_small_delta():
    out = v1_verdict([0.8530, 0.8536], reference=0.8534, targets_equal=True)
    assert out["auc_mean"] == pytest.approx(0.8533)
    assert out["auc_std"] == pytest.approx(0.0003)          # population std, like the pilot's macros.std()
    assert out["reference"] == 0.8534
    assert out["delta"] == pytest.approx(-0.0001)
    assert out["pass"] is True
    assert v1_verdict([0.8533], reference=0.8534, targets_equal=False)["pass"] is False
    assert v1_verdict([0.8600], reference=0.8534, targets_equal=True)["pass"] is False
    assert v1_verdict([0.8470], reference=0.8534, targets_equal=True)["pass"] is False


def test_v1_verdict_tolerance_boundary_is_inclusive_despite_float_error():
    # 0.8564 - 0.8534 == 0.003000000000000002 in binary floating point
    assert v1_verdict([0.8564], reference=0.8534, targets_equal=True)["pass"] is True
    assert v1_verdict([0.8565], reference=0.8534, targets_equal=True)["pass"] is False


def test_classify_label_diffs_separates_tie_breaking_from_other_differences():
    n_topics = 3
    # rows: k=4 neighbour labels, nearest first
    neighbor_labels = np.array([
        [2, 0, 0, 2],   # tie 0/2 (2 votes each); nearest-among-tied = 2, lowest index = 0
        [1, 1, 1, 0],   # clear winner 1
        [0, 2, 1, 1],   # clear winner 1
        [1, 0, 1, 0],   # tie 0/1; nearest = 1, lowest = 0
    ])
    pilot = np.array([2, 1, 1, 1])      # nearest-among-tied rule
    harness = np.array([0, 1, 2, 1])    # row 0: lowest-index tie-break; row 2: a non-tie disagreement
    out = classify_label_diffs(neighbor_labels, pilot, harness, n_topics)
    assert out == {
        "n_rows": 4, "n_tied": 2, "n_diff": 2, "n_diff_tied": 1,
        "n_diff_pilot_is_nearest_tied": 1, "n_diff_harness_is_lowest_index": 1,
    }
    assert all(type(v) is int for v in out.values())


def test_script_import_is_light():
    """Heavy imports (torch, harness, pilots) must be lazy so this test file
    imports the script without data or a GPU."""
    here = Path(__file__).resolve().parent
    code = ("import sys; sys.path.insert(0, %r); import validate_v1_stage2; "
            "heavy = [m for m in ('torch', 'sklearn', 'scripts.buddy_percept_sweep.h2h_eval') if m in sys.modules]; "
            "assert not heavy, heavy" % str(here))
    subprocess.run([sys.executable, "-c", code], check=True)
