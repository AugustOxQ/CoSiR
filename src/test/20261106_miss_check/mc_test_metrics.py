"""Differential AP tests without requesting any pair computation from r23."""
import numpy as np
import pytest

from mc_metrics import evaluate
from tcl_metrics import evaluate as reference


@pytest.mark.parametrize('readout', ['r0', 'center', 'center_k10'])
def test_label_only_self_exclusion_and_crossed_missing_labels(readout):
    rng = np.random.default_rng(24)
    x = rng.normal(size=(31, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    means = (np.zeros(8, dtype=np.float32),) * 2
    a = np.arange(len(x), dtype=np.int64) % 3
    b = np.arange(len(x), dtype=np.int64) % 4
    b[::5] = -1
    labels = {'A': dict(q=a, g=a), 'B': dict(q=b, g=b)}
    indices = np.asarray([0, 5, 13, 30], dtype=np.int64)
    selected = {k: dict(q=v['q'][indices], g=v['g']) for k, v in labels.items()}
    got = evaluate(x[indices], x, selected, True, readout, 'cpu', means=means, self_indices=indices)
    # Removing the self gallery row lets r23 evaluate with pid=None and
    # self_match=False: it computes no first-hit ranks in this invocation.
    for j, i in enumerate(indices):
        keep = np.arange(len(x)) != i
        entries = {k: dict(q=v['q'][i:i + 1], g=v['g'][keep]) for k, v in labels.items()}
        ref = reference(x[i:i + 1], x[keep], None, None, entries, False, readout, 'cpu', means=means)['ap']
        for k in got:
            np.testing.assert_allclose(got[k][j], ref[k][0], atol=1e-6, rtol=0, equal_nan=True)
