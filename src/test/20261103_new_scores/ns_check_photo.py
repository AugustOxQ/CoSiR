"""r17 ticket 04 check (not a test file): `ns_photo` rows and r10's route reproduce stored r14 AMI at 20 values.

Compared (seed 0, mean of the three k-means states, `ac_cluster.partitions` = `rs_vh.r20_partition`, 1 thread):
  val   Affection photo model, siglip2 / siglip1 / clip_l14: r14 `stage1/affection_ranges_<enc>.json`
        ranges_doc.clu.per_seed[0] (validation AMI at 20 of the five final runs).
  val   EmoSet photo model, qwen: r14 `records/stage2_emoset.json` ranges.clu.per_seed[0].
  val   Affection photo model, qwen (r13's): r13 `src/test/20261031_photo_finetune/stage2_records.json`
        ranges.clu.per_seed[0].
  test  Affection photo model, siglip2, seed 0: r14's cached partitions `stage3/partitions/affection__test__photo_siglip2__s0.npz`
        (array equality) and their AMI.
Usage: python ns_check_photo.py [val|test|all]. Tolerance `ns_paths.TOL_REPLAY`. Writes nothing.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as NP  # noqa: E402
import ns_photo as P  # noqa: E402
import ac_cluster  # noqa: E402


def val_checks():
    rows = []
    for enc in ('siglip2', 'siglip1', 'clip_l14'):
        ref = json.loads((NP.R14_OUT / f'stage1/affection_ranges_{enc}.json').read_text())['ranges_doc']['clu']['per_seed'][0]
        rows.append(('affection', enc, 'val', ref))
    r13 = Path('/project/CoSiR/src/test/20261031_photo_finetune/stage2_records.json')
    rows.append(('affection', 'qwen', 'val', json.loads(r13.read_text())['ranges']['clu']['per_seed'][0]))
    ref = json.loads(P.R14_STAGE2_EMOSET.read_text())['ranges']['clu']['per_seed'][0]
    rows.append(('emoset', 'qwen', 'val', ref))
    out = []
    for set_name, enc, split, ref in rows:
        y = P.labels(set_name, split)
        parts = ac_cluster.partitions(P.photo(set_name, split, enc, 0))
        got = float(ac_cluster.ami(parts, y))
        out.append((f'{set_name}/{enc}/{split}/s0', got, ref, abs(got - ref), True))
    return out


def test_split_checks():
    enc, seed = 'siglip2', 0
    y = P.labels('affection', 'test')
    parts = ac_cluster.partitions(P.photo('affection', 'test', enc, seed))
    path = NP.R14_OUT / f'stage3/partitions/affection__test__photo_{enc}__s{seed}.npz'
    with np.load(path) as z:
        stored = z[z.files[0]] if len(z.files) == 1 else z['labels']
    same = bool(np.array_equal(stored, parts))
    got, ref = float(ac_cluster.ami(parts, y)), float(ac_cluster.ami(stored, y))
    out = [(f'affection/{enc}/test/s0 (cached partitions equal: {same})', got, ref, abs(got - ref), same)]
    # plain (r14 frozen_siglip2) and teacher (r12 teacher_artwork) rows against their stored test partitions
    for name, x, path in (('plain siglip2', P.plain('affection', 'test', 'siglip2'),
                           NP.R14_OUT / 'stage3/partitions/affection__test__frozen_siglip2__s-.npz'),
                          ('teacher artwork', P.teacher('affection', 'test', 'qwen'),
                           NP.R12_OUT / 'partitions/affection__test__teacher_artwork__s-.npz')):
        parts = ac_cluster.partitions(x)
        with np.load(path) as z:
            stored = z['labels']
        same = bool(np.array_equal(stored, parts))
        got, ref = float(ac_cluster.ami(parts, y)), float(ac_cluster.ami(stored, y))
        out.append((f'affection/{name}/test (cached partitions equal: {same})', got, ref, abs(got - ref), same))
    return out


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    res = (val_checks() if which in ('val', 'all') else []) + (test_split_checks() if which in ('test', 'all') else [])
    for name, got, ref, diff, _same in res:
        print(f'{name}: recomputed {got!r}  stored {ref!r}  max abs diff {diff:.3e}  {"OK" if diff <= NP.TOL_REPLAY else "FAIL"}')
    assert all(r[3] <= NP.TOL_REPLAY for r in res), 'replay differs beyond TOL_REPLAY'
    assert all(r[4] for r in res), 'recomputed partitions differ from the cached ones'
