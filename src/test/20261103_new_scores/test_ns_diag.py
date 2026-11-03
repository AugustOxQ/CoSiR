"""Stage 3 regressions with stubbed data and temporary outputs only."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_diag  # noqa: E402
import ns_paths as P  # noqa: E402
import ns_picks  # noqa: E402
import ns_ret  # noqa: E402


@pytest.fixture
def outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'OUT', tmp_path)
    monkeypatch.setattr(P, 'SMOKE', 1)
    return tmp_path


@pytest.mark.parametrize('missing_pick', [False, True])
def test_r0_rows_scores_held_arms_and_keeps_validation_fields(outputs, monkeypatch, missing_pick):
    record = {ns_picks.key_of(*key): dict(pick='center+k3', scores=dict(
        r0_map={a: 0.1 for a in P.ASPECTS}, map={a: 0.2 for a in P.ASPECTS}),
        unit={a: 0.01 for a in P.ASPECTS}) for key in ns_picks.PICKS}
    missing = ns_picks.key_of(*ns_picks.PICKS[0])
    if missing_pick:
        del record[missing]
    raw = np.array([[2., 1.], [1., 3.], [4., 2.], [2., 5.]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    label_calls, calls = [], []

    def get_labels(set_, split, aspect):
        label_calls.append((set_, split, aspect))
        assert split == P.HELD_SPLIT[set_]
        return labels

    monkeypatch.setattr(ns_diag.ns_data, 'labels', get_labels)
    monkeypatch.setattr(ns_diag.ns_data, 'untold', lambda s, split, e, seed: raw + np.float32(seed))
    monkeypatch.setattr(ns_diag.ns_data, 'percept_probs', lambda s, split, e, seed: raw + np.float32(seed))
    monkeypatch.setattr(ns_diag.ns_data, 'crl', lambda s, split, a: raw)

    def arm_ap(get_val, get_eval, pick, y, seeds, workers=8):
        assert pick == 'none+k0' and y is labels
        xs = [get_eval(s) for s in (P.SEEDS if seeds else (None,))]
        assert all(np.array_equal(get_val(s), x) for s, x in zip(P.SEEDS if seeds else (None,), xs))
        calls.append((seeds, xs))
        ap = np.array([x.sum(axis=1) / 20 for x in xs], dtype=np.float64)
        if seeds is None:
            ap = ap[0]
        return dict(ap=ap, point=-99., per_seed=ap.mean(axis=1).tolist() if seeds else None,
                    rows=np.arange(4), dropped=0)

    monkeypatch.setattr(ns_ret, 'arm_ap', arm_ap)
    doc = ns_diag.r0_rows(record)
    assert doc['missing'] == ([missing] if missing_pick else [])
    assert [r['key'] for r in doc['rows']] == list(record)
    assert len(calls) == len(record) * len(P.ASPECTS) == len(label_calls)
    for row in doc['rows']:
        arm, set_, enc = row['key'].split('/')
        assert row['split'] == P.HELD_SPLIT[set_]
        assert {k: row[k] for k in ('pick', 'r0_map', 'map_at_pick', 'unit')} == dict(
            pick=record[row['key']]['pick'], r0_map=record[row['key']]['scores']['r0_map'],
            map_at_pick=record[row['key']]['scores']['map'], unit=record[row['key']]['unit'])
        seeds = None if arm == 'crl' else P.SEEDS
        expected = [raw] if seeds is None else [raw + np.float32(s) for s in seeds]
        if arm == 'matched':
            expected = [ns_ret._unit(x) for x in expected]
        maps = np.array([x.sum(axis=1) / 20 for x in expected], dtype=np.float64)
        for aspect in P.ASPECTS:
            assert row['aspects'][aspect] == dict(mean=float(maps.mean()),
                per_seed=maps.mean(axis=1).tolist() if seeds else None)
        for _ in P.ASPECTS:
            got_seeds, xs = calls.pop(0)
            assert got_seeds == seeds
            assert all(np.array_equal(x, want) for x, want in zip(xs, expected))
    assert json.loads((outputs / 'stage3/r0_rows.json').read_text()) == doc


@pytest.mark.parametrize('command, argv', [
    ('sweep', ['sweep', '--set', 'mmart', '--enc', 'qwen']),
    ('figure', ['figure', '--set', 'mmart']),
    ('per_class', ['per-class', '--set', 'mmart', '--enc', 'qwen']),
    ('told_untold', ['told-untold', '--set', 'mmart', '--enc', 'qwen']),
    ('qwen_crl', ['qwen-crl']), ('r0_rows', ['r0']), ('tables', ['tables', '--set', 'mmart']),
])
@pytest.mark.parametrize('smoke, verdicts_exist', [(False, False), (False, True), (True, False)])
def test_main_requires_verdicts_before_stage3(outputs, monkeypatch, command, argv, smoke, verdicts_exist):
    monkeypatch.setattr(P, 'SMOKE', smoke)
    monkeypatch.setattr(P, 'RECORDS', outputs / 'committed')
    verdicts = P.record_path('verdicts.json')
    if verdicts_exist:
        P.write_json(verdicts, {})
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(kmeans_runs=0, seconds=0, classes=[], rows=[], missing=[],
                    class_count_beside_ami20=[], seed_spreads=[])

    monkeypatch.setattr(ns_diag, command, run)
    if not smoke and not verdicts_exist:
        with pytest.raises(AssertionError, match='verdicts.json'):
            ns_diag.main(argv)
        assert not calls
    else:
        ns_diag.main(argv)
        assert len(calls) == 1
