"""Paper topic cuts and Stage 1 collapse judgments under rule §4 R-COL.

Ticket 07 artifact contract, under outputs/percept/paper/stage1/:
  <version>__s<seed>.npz, versions clip_roberta/qwen_qwen, seeds 0..4.
  center_norms: (100,) finite nonnegative norms in original center order.
  fit_counts: (100,) nonnegative integers, before pruning, sum 67,327.
  kept_strict, kept_k40: (100,) bool masks in original center order.
  fit_topics_strict, fit_topics_k40: (67,327,) integer hard-topic IDs on
    fit pairs only, indexing np.flatnonzero(kept_<cut>) in ascending order.
  Other arrays (caption topics, q_paint_eval, etc.) may coexist in the NPZ.

The JSON has runs (version, seed, cut, kept_count, share, collapsed, artifact,
array_sha256) and entries (B2..B5 -> eligible, seeds). Each hash is SHA-256
of the saved array's C-order bytes, matching pc_stage1.array_sha256. Both
readings of an entry share its seed list. B1 has no collapse rule. V4's
non-finite held-output check belongs to the scorer; this module reads no held
outputs. Defaults read artifacts from the main checkout and write here.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE1 = Path('/project/CoSiR/src/test/20261020_claim_test/outputs/percept/paper/stage1')
OUTPUT = HERE / 'outputs/percept/rcol_stage1.json'
sys.path.insert(0, str(HERE.parent / '20261018_percept_core'))
import pc_fixes


def strict_mask(center_norms):
    """Algorithm 1, gamma=2, in original center order."""
    assert center_norms.shape == (100,)
    return pc_fixes.kept_centers(center_norms, gamma=2)


def k40_mask(fit_counts):
    """A3's forty most-used centers, with core's tie rule."""
    assert fit_counts.shape == (100,)
    return pc_fixes.top_used(fit_counts, 40)


def judge(cut, kept, fit_topics):
    """Judge one cut; hard topics are local IDs over kept centers, on fit pairs only."""
    assert cut in ('strict', 'k40')
    assert kept.shape == (100,) and kept.dtype == np.bool_
    count = int(kept.sum())
    assert count > 0 and (cut != 'k40' or count == 40)
    assert fit_topics.ndim == 1 and len(fit_topics) > 0
    assert np.issubdtype(fit_topics.dtype, np.integer)
    assert ((fit_topics >= 0) & (fit_topics < count)).all()
    share = float(np.bincount(fit_topics, minlength=count).max() / len(fit_topics))
    collapsed = share > .5 or (cut == 'strict' and (count < 10 or count > 90))
    return dict(kept_count=count, share=share, collapsed=collapsed)


def eligible_entries(runs):
    """B2..B5 seed lists, shared by own/r20; ticket 03 additionally applies V4."""
    entries = {}
    for entry, version, cut in (
            ('B2', 'clip_roberta', 'strict'), ('B3', 'qwen_qwen', 'strict'),
            ('B4', 'clip_roberta', 'k40'), ('B5', 'qwen_qwen', 'k40')):
        selected = [r for r in runs if r['version'] == version and r['cut'] == cut]
        assert sorted(r['seed'] for r in selected) == list(range(5))
        seeds = sorted(r['seed'] for r in selected if not r['collapsed'])
        entries[entry] = dict(eligible=bool(seeds), seeds=seeds)
    return entries


def array_sha256(array):
    """Hash C-order array bytes, preserving the saved dtype (core's convention)."""
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def read_stage1(path):
    """Read one paper run and return its strict and k40 judgments."""
    path = Path(path)
    version, seed = path.stem.split('__s')
    assert version in ('clip_roberta', 'qwen_qwen') and int(seed) in range(5)
    runs = []
    with np.load(path, allow_pickle=False) as z:
        required = {'center_norms', 'fit_counts', 'kept_strict', 'kept_k40',
                    'fit_topics_strict', 'fit_topics_k40'}
        assert required <= set(z.files), f'{path}: missing {required - set(z.files)}'
        norms, counts = z['center_norms'], z['fit_counts']
        expected_masks = dict(strict=strict_mask(norms), k40=k40_mask(counts))
        assert counts.sum() == 67327, f'{path}: expected 67,327 fit pairs'
        for cut in ('strict', 'k40'):
            arrays = {key: z[key] for key in (
                'center_norms', 'fit_counts', f'kept_{cut}', f'fit_topics_{cut}')}
            assert arrays[f'fit_topics_{cut}'].shape == (67327,)
            assert np.array_equal(arrays[f'kept_{cut}'], expected_masks[cut]), \
                f'{path}: {cut} mask differs from the rule'
            result = judge(cut, arrays[f'kept_{cut}'], arrays[f'fit_topics_{cut}'])
            runs.append(dict(version=version, seed=int(seed), cut=cut, **result,
                             artifact=str(path.resolve()),
                             array_sha256={k: array_sha256(v) for k, v in arrays.items()}))
    return runs


def write_stage1(stage1_dir=STAGE1, output=OUTPUT):
    """Atomically save all ten paper runs' R-COL results before held scoring."""
    runs = []
    for version in ('clip_roberta', 'qwen_qwen'):
        for seed in range(5):
            runs.extend(read_stage1(Path(stage1_dir) / f'{version}__s{seed}.npz'))
    report = dict(runs=runs, entries=eligible_entries(runs))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                     prefix=output.stem + '.', suffix='.pending.json',
                                     delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            os.replace(pending, output)
        finally:
            pending.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1-dir', type=Path, default=STAGE1,
                        help=f'Paper Stage 1 NPZ directory (default: {STAGE1})')
    parser.add_argument('--output', type=Path, default=OUTPUT,
                        help=f'R-COL JSON path (default: {OUTPUT})')
    args = parser.parse_args()
    write_stage1(args.stage1_dir, args.output)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
