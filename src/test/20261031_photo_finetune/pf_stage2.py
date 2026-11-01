"""Photo-model seed ranges and validation read-out (r13 spec §§3.4, 6; D10).

Frozen r12/r11 helpers are imported unchanged. Callbacks and array mappings use
run indices 0..4 (training seeds 100..104 belong to the runner, not this module).
"""
import json
from pathlib import Path

import numpy as np

import pf_paths  # puts the frozen rounds on sys.path
import ac_stats
import ac_select
import rt_readout


def _check_arrays(xs, labels, dtype):
    """Check the array boundary without casting or repairing unit rows."""
    assert isinstance(labels, np.ndarray) and labels.dtype == np.int64 and labels.ndim == 1
    assert (labels >= 0).all() and (labels < len(pf_paths.EMOTIONS)).all()
    shapes = set()
    for x in xs.values():
        assert isinstance(x, np.ndarray) and x.dtype == dtype and x.ndim == 2
        assert len(x) == len(labels) and x.shape[1] > 0, (x.shape, labels.shape)
        assert np.isfinite(x).all()
        assert np.max(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1)) < pf_paths.TOL['unit_norm']
        shapes.add(x.shape)
    assert len(shapes) == 1, 'one shape across five runs'
    return shapes.pop()


def ranges(get_stored, get_embed, *, labels=None, dist=None, workers=8):
    """Measure five validation runs, preserving stored rows for clustering.

    get_stored(s): float16 [n, width]; get_embed(s): unit float32 [n, width],
    both in sorted validation-key order. Labels/distributions default to r12's
    full validation data; explicit arrays allow synthetic checks.
    """
    try:
        stored = {s: get_stored(s) for s in pf_paths.SEEDS}
        embedded = {s: get_embed(s) for s in pf_paths.SEEDS}
    except (KeyError, IndexError) as exc:
        raise AssertionError('five runs (indices 0..4) are required') from exc
    if labels is None or dist is None:
        import ac_data
        labels = ac_data.labels('val') if labels is None else labels
        dist = ac_data.emotion_dist('val') if dist is None else dist
    assert _check_arrays(stored, labels, np.float16) == _check_arrays(embedded, labels, np.float32)
    assert isinstance(dist, np.ndarray) and dist.dtype == np.float64
    assert dist.shape == (len(labels), len(pf_paths.EMOTIONS))
    assert np.isfinite(dist).all() and (dist >= 0).all()
    assert np.allclose(dist.sum(axis=1), 1, rtol=0, atol=1e-12)
    return ac_stats.measure_seed_ranges(embedded.__getitem__, stored.__getitem__, labels, dist, workers=workers)


def readout(xs_val, r_map, *, labels=None, keys=None, workers=8):
    """Return {pick: name, scores: table} using r'_mAP and r11's full menu.

    xs_val maps run index 0..4 to unit float32 validation arrays in sorted-key
    order. Cross-fit, the guard and the 0.001-r tie rule are r12's unchanged.
    Explicit labels/keys support synthetic arrays; otherwise use full val data.
    """
    assert set(xs_val) == set(pf_paths.SEEDS), 'exactly five runs (indices 0..4) are required'
    assert np.isfinite(r_map) and r_map > 0, 'r_map must be finite positive'
    if labels is None:
        import ac_data
        labels = ac_data.labels('val')
        keys = ac_data.keys('val') if keys is None else keys
    _check_arrays(xs_val, labels, np.float32)
    if keys is not None:
        keys = list(keys)
        assert len(set(keys)) == len(keys), 'one unique key per validation row'
    table, pick = ac_select.arm_table({('emotion', s): x for s, x in xs_val.items()},
                                      {'emotion': labels}, {'emotion': r_map}, keys=keys, workers=workers)
    return {'pick': rt_readout.name(pick), 'scores': table}


def record(seed_ranges, selected_readout, *, r12_ranges=None):
    """Build the shared stage-2 record, unrounded; ticket 10 writes/commits it.

    selected_readout is readout()'s result. The default comparison ranges are
    read from frozen r12; an explicit document supports isolated tests.
    """
    if r12_ranges is None:
        r12_ranges = ac_stats.load_ranges(pf_paths.R12_OUT / 'stage0b/ranges.json')
    for doc in (seed_ranges, r12_ranges):
        assert set(doc) == {'clu', 'map', 'ndcg'}
        checked = ac_stats.ranges_doc(*(doc[m]['per_seed'] for m in ('clu', 'map', 'ndcg')))
        assert doc == checked, 'ranges must equal the unrounded max minus min over five runs'
    assert set(selected_readout) == {'pick', 'scores'}
    assert selected_readout['pick'] in selected_readout['scores']
    return json.loads(json.dumps({'ranges': seed_ranges, 'r12_ranges': r12_ranges,
                                  'readout': selected_readout}, allow_nan=False))


def gate_ii_stage2(*, outputs=None, workers=8):
    """Replay r12's ranges and all c* read-outs; write stage0a/stage2_replay.json.

    Plain callable for ticket 10. CPU-only, controller-run: approximately 1–3
    minutes based on r12's stage0b (28 s) and whole stage1 (306 s). Every number
    is compared at absolute tolerance 1e-9; names, picks and flags match exactly.
    Returns gate evidence (including failures), never writes frozen outputs.
    """
    import ac_data

    reference_ranges = ac_stats.load_ranges(pf_paths.R12_OUT / 'stage0b/ranges.json')
    reference_scores = json.loads((pf_paths.R12_OUT / 'stage1/scores.json').read_text())['readout']['affection']['cstar']
    reference_pick = json.loads((pf_paths.R12_OUT / 'stage1/picks.json').read_text())['affection']['readout']['cstar']
    measured = ranges(lambda s: ac_data.stored('cstar', 'val', s, head='emotion'),
                      lambda s: ac_data.embed('cstar', 'val', s, head='emotion'), workers=workers)
    selected = readout({s: ac_data.embed('cstar', 'val', s, head='emotion') for s in pf_paths.SEEDS},
                       reference_ranges['map']['r'], workers=workers)
    items = {}

    def compare(name, value, reference, *, exact=False):
        ok = value == reference if exact else abs(value - reference) <= pf_paths.TOL['ii']
        items[name] = {'value': value, 'reference': reference, 'ok': bool(ok)}

    for metric in ('clu', 'map', 'ndcg'):
        for s in pf_paths.SEEDS:
            compare(f'ranges.{metric}.per_seed.{s}', measured[metric]['per_seed'][s],
                    reference_ranges[metric]['per_seed'][s])
        compare(f'ranges.{metric}.r', measured[metric]['r'], reference_ranges[metric]['r'])
    compare('readout.names', sorted(selected['scores']), sorted(reference_scores), exact=True)
    compare('readout.pick', selected['pick'], reference_pick, exact=True)
    for name, expected in reference_scores.items():
        actual = selected['scores'].get(name)
        if actual is None:  # names evidence already fails; keep the absent row visible
            items[f'readout.{name}'] = {'value': None, 'reference': expected, 'ok': False}
            continue
        compare(f'readout.{name}.map.emotion', actual['map']['emotion'], expected['map']['emotion'])
        compare(f'readout.{name}.score', actual['score'], expected['score'])
        compare(f'readout.{name}.kept', actual['kept'], expected['kept'], exact=True)
    evidence = {'items': items, 'ok': all(item['ok'] for item in items.values())}
    target = Path(pf_paths.OUTPUTS if outputs is None else outputs) / 'stage0a/stage2_replay.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=1, allow_nan=False) + '\n')
    return evidence
