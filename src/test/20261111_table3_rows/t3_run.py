"""Table 3 rows, run and gates (round r27). Subcommands (all take --workers N, default 8):

    gate-tpips      TPIPS on ArtELingo through the generic unseeded path against records/labels.json (SC2 .../tpips cells)
    gate-percept    r17's matched PercepT on CLIP L/14 through the generic seeded path against r17's records
    vlm2vec         VLM2Vec-V2 row: unseeded, pick on validation (unit: Qwen c*'s ranges, as TPIPS), test mAP
    percept-clip    PercepT on CLIP L/14 row (r26's probabilities), kept seeds, pick on validation, test mAP

The code's name for the test split is `held`; printed text says "test". Inputs are read only; outputs go to out/.
Gates pass at max abs difference <= 1e-9.

    python t3_run.py gate-tpips
    python t3_run.py gate-percept
    python t3_run.py vlm2vec --test-root /data/SSD2/pre_extract/artelingo/vlm2vec_v2 --val-root /data/SSD2/pre_extract/artelingo/vlm2vec_v2_val
    python t3_run.py percept-clip --seeds 0,1,2,3,4 --val-dir <r26 val/k40> --test-dir <r26 test dir> [--file-pattern 'L1__s{seed}.npz']
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t3_core as C  # noqa: E402
import t3_inputs as I  # noqa: E402

TOL = 1e-9
N_VAL, N_TEST = 6152, 12281
DEFAULT_VLM_TEST = '/data/SSD2/pre_extract/artelingo/vlm2vec_v2'
DEFAULT_VLM_VAL = '/data/SSD2/pre_extract/artelingo/vlm2vec_v2_val'
DEFAULT_PERCEPT_VAL = '/data/SSD2/cosir_r26/20261111_percept_original/outputs/val/k40'
DEFAULT_PATTERN = 'L1__s{seed}.npz'
TPIPS_CELLS = C.OM_ROUND / 'records' / 'labels.json'
TPIPS_PICKS = C.OM_ROUND / 'records' / 'stage1_om_picks.json'
R17_PICKS = C.ns_paths.RECORDS / 'stage1_picks.json'
R17_VERDICTS = C.ns_paths.RECORDS / 'verdicts.json'


def _memo(fn):
    cache = {}

    def get(*key):
        if key not in cache:
            cache[key] = fn(*key)
        return cache[key]
    return get


def _readout_row(res):
    return dict(pick=res['pick'], readout=res['readout'], validation_table=res['table'],
                per_seed_validation_map=res['per_seed_map'], dropped_validation=res['dropped'])


def _arm(get_val, get_test, unit, seeds, workers):
    """Pick on validation, then test mAP at the pick: the whole generic path. Returns (pick doc, test dict)."""
    C.check_validation_keys()
    lv, lt = C.labels(C.DEV_SPLIT), C.labels(C.TEST_SPLIT)
    t0 = time.monotonic()
    pk = C.pick_readout(get_val, lv, unit, seeds, workers)
    print(f'  pick {pk["pick"]} ({time.monotonic() - t0:.0f}s)', flush=True)
    test = C.score_test(get_val, get_test, pk['pick'], lt, seeds, workers)
    print(f'  test mAP ' + ' '.join(f'{a} {test[a]["map"]:.6f}' for a in C.ASPECTS) + f' ({time.monotonic() - t0:.0f}s)', flush=True)
    return pk, test


def _item(name, ours, theirs):
    return dict(name=name, ours=float(ours), theirs=float(theirs), abs_diff=abs(float(ours) - float(theirs)))


def _gate_result(items, pick_ok, name, **extra):
    worst = max(i['abs_diff'] for i in items)
    ok = bool(pick_ok and worst <= TOL)
    print(f'{name}: pass={ok} pick_matches_record={pick_ok} max_abs_diff={worst:.3e} (tol {TOL:g}, {len(items)} values)')
    for i in items:
        print(f'  {i["name"]}: ours {i["ours"]!r} record {i["theirs"]!r} diff {i["abs_diff"]:.3e}')
    return {'pass': ok, 'pick_matches_record': bool(pick_ok), 'max_abs_diff': float(worst), 'tol': TOL, 'items': items, **extra}


# ================================================================================================ gates
def gate_tpips(workers):
    import om_data
    unit, source = C.tpips_unit()
    val, test = _memo(lambda a, _s: om_data.tpips(C.SET, C.DEV_SPLIT, a)), _memo(lambda a, _s: om_data.tpips(C.SET, C.TEST_SPLIT, a))
    pk, res = _arm(val, test, unit, None, workers)
    stored_pick = json.loads(TPIPS_PICKS.read_text())[f'tpips/{C.SET}']
    cells = {c['id']: c for c in json.loads(TPIPS_CELLS.read_text())['cells']}
    items = []
    for a in C.ASPECTS:
        items.append(_item(f'SC2/{C.SET}/qwen/{a}/map/-/tpips test mAP', res[a]['map'], cells[f'SC2/{C.SET}/qwen/{a}/map/-/tpips']['comp']))
        items.append(_item(f'tpips/{C.SET} validation mAP at the pick, {a}', pk['table'][pk['pick']]['map'][a], stored_pick['scores']['map'][a]))
    pick_ok = pk['pick'] == stored_pick['pick'] == 'none+k10'
    out = _gate_result(items, pick_ok, 'gate-tpips', pick=pk['pick'], record_pick=stored_pick['pick'], unit=unit, unit_source=source)
    C.write_json(C.OUT / 'gate_tpips.json', out)
    return out


def gate_percept(workers):
    import ns_data
    unit, source = C.clip_l14_unit()
    seeds = C.ALL_SEEDS
    val = _memo(lambda a, s: C.ns_ret._percept_rows(C.SET, C.DEV_SPLIT, 'clip_l14', s))      # r17's own rows, per seed
    test = _memo(lambda a, s: C.ns_ret._percept_rows(C.SET, C.TEST_SPLIT, 'clip_l14', s))
    pk, res = _arm(val, test, unit, seeds, workers)
    stored_pick = json.loads(R17_PICKS.read_text())[f'matched/{C.SET}/clip_l14']
    cells = {c['id']: c for c in json.loads(R17_VERDICTS.read_text())['cells']}
    items = []
    for a in C.ASPECTS:
        items.append(_item(f'SC8/{C.SET}/clip_l14/{a}/map/-/matched test mAP', res[a]['map'], cells[f'SC8/{C.SET}/clip_l14/{a}/map/-/matched']['comp']))
        items.append(_item(f'matched/{C.SET}/clip_l14 validation mAP at the pick, {a}', pk['table'][pk['pick']]['map'][a], stored_pick['scores']['map'][a]))
    pick_ok = pk['pick'] == stored_pick['pick']
    out = _gate_result(items, pick_ok, 'gate-percept', pick=pk['pick'], record_pick=stored_pick['pick'], unit=unit, unit_source=source,
                       seeds=list(seeds))
    C.write_json(C.OUT / 'gate_percept.json', out)
    return out


# ================================================================================================ the two rows
def _record(name, pk, test, unit, source, seeds, inputs, extra):
    return dict(
        row=name, set=C.SET, test_split_name_in_code=C.TEST_SPLIT, metric='image-to-image label mAP on the ArtELingo test split',
        pick=pk['pick'], readout=pk['readout'], validation_table=pk['table'], per_seed_validation_map=pk['per_seed_map'],
        dropped_validation=pk['dropped'], seeds=None if seeds is None else list(seeds),
        test_map={a: test[a]['map'] for a in C.ASPECTS},
        test_map_per_seed={a: test[a]['per_seed'] for a in C.ASPECTS},
        n_queries={a: test[a]['n_queries'] for a in C.ASPECTS}, dropped_test={a: test[a]['dropped'] for a in C.ASPECTS},
        n_validation_rows=N_VAL, n_test_rows=N_TEST, unit=unit, unit_source=source, inputs=inputs, git_head=C.git_head(), **extra)


def run_vlm2vec(args):
    unit, source = C.tpips_unit()
    vs, ts = I.Vlm2VecStore(args.val_root, C.DEV_SPLIT, N_VAL), I.Vlm2VecStore(args.test_root, C.TEST_SPLIT, N_TEST)
    pk, test = _arm(lambda a, _s: vs.rows(a), lambda a, _s: ts.rows(a), unit, None, args.workers)
    prov = dict(validation=vs.provenance(), test=ts.provenance())
    for a in C.ASPECTS:
        assert prov['validation'][a]['instruction'] == prov['test'][a]['instruction'], f'{a}: instructions differ between splits'
    rec = _record('VLM2Vec-V2', pk, test, unit,
                  source + ' (Qwen c* validation mAP ranges, as TPIPS: the unseeded arm keeps its unit)', None, prov,
                  dict(arm_note='one embedding per aspect instruction; the aspect\'s own embedding scores that aspect; unseeded'))
    path = C.write_json(C.OUT / 'vlm2vec.json', rec)
    print(f'vlm2vec: pick {pk["pick"]}; test mAP ' + ' '.join(f'{a} {test[a]["map"]:.6f}' for a in C.ASPECTS) + f' -> {path}')
    return rec


def run_percept_clip(args):
    seeds = C.parse_seeds(args.seeds)
    unit, source = C.clip_l14_unit()
    vf = I.PerceptFiles(args.val_dir, args.file_pattern, C.DEV_SPLIT, seeds, N_VAL)
    tf = I.PerceptFiles(args.test_dir, args.test_file_pattern or args.file_pattern, C.TEST_SPLIT, seeds, N_TEST)
    pk, test = _arm(lambda a, s: vf.rows(s), lambda a, s: tf.rows(s), unit, seeds, args.workers)
    prov = dict(validation=vf.provenance(), test=tf.provenance())
    rec = _record('PercepT on CLIP L/14', pk, test, unit,
                  source + ' (CLIP L/14 c* validation mAP ranges, as r17\'s matched/artelingo/clip_l14 pick)', seeds, prov,
                  dict(arm_note='topic probabilities (40) as rows, each divided by its float32 norm; pick = cross-fit mean over '
                                'the kept seeds; test mAP = mean over the kept seeds, each seed scored through ns_ret.arm_ap '
                                '(seeds=None) and averaged'))
    path = C.write_json(C.OUT / 'percept_clip.json', rec)
    print(f'percept-clip seeds {list(seeds)}: pick {pk["pick"]}; test mAP ' + ' '.join(f'{a} {test[a]["map"]:.6f}' for a in C.ASPECTS)
          + f' -> {path}')
    return rec


# ================================================================================================ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def add(name, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument('--workers', type=int, default=8, choices=range(1, 9))
        return s
    add('gate-tpips', 'TPIPS through the generic unseeded path against the stored cells')
    add('gate-percept', 'r17 matched PercepT (CLIP L/14) through the generic seeded path against r17 records')
    s = add('vlm2vec', 'VLM2Vec-V2 row')
    s.add_argument('--test-root', default=DEFAULT_VLM_TEST)
    s.add_argument('--val-root', default=DEFAULT_VLM_VAL)
    s = add('percept-clip', 'PercepT on CLIP L/14 row (r26 probabilities)')
    s.add_argument('--seeds', required=True, help='kept seeds, e.g. 0,1,3 (from r26\'s collapse record)')
    s.add_argument('--val-dir', default=DEFAULT_PERCEPT_VAL)
    s.add_argument('--test-dir', required=True)
    s.add_argument('--file-pattern', default=DEFAULT_PATTERN, help='file name with {seed}; used for validation (and test)')
    s.add_argument('--test-file-pattern', default=None, help='test file name with {seed}, if it differs')
    args = p.parse_args(argv)
    t0 = time.monotonic()
    {'gate-tpips': lambda: gate_tpips(args.workers), 'gate-percept': lambda: gate_percept(args.workers),
     'vlm2vec': lambda: run_vlm2vec(args), 'percept-clip': lambda: run_percept_clip(args)}[args.cmd]()
    print(f'{args.cmd}: {time.monotonic() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
