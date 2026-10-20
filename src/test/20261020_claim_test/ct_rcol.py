"""Paper topic cuts and Stage 1 collapse judgments under rule §4 R-COL.

Ticket 07 artifact contract, under outputs/percept/paper/stage1/:
  <version>__s<seed>.npz, versions clip_roberta/qwen_qwen, seeds 0..4.
  center_norms: (100,) finite nonnegative norms in original center order.
  centers: (100, 128) float32; center_norms matches their row norms (rtol 1e-6).
  fit_latent: (67,327, 128) float32, in fit-row order.
  fit_counts: (100,) nonnegative integers, sum 67,327: per-fit-pair argmax
    of Student-t Q (alpha=1) over all 100 centers, lowest index on a tie.
  kept_strict, kept_k40: (100,) bool masks in original center order.
  fit_topics_strict, fit_topics_k40: (67,327,) integer hard-topic IDs on
    fit pairs only: argmax of Q renormalized over the kept centers (equivalently
    the kept columns), lowest index on a tie, as pc_stage1.assign_captions does;
    IDs index np.flatnonzero(kept_<cut>) in ascending order.
  Other arrays (caption topics, q_paint_eval, etc.) may coexist in the NPZ.

The JSON has runs (version, seed, cut, kept_count, share, collapsed, artifact,
array_sha256, fit_counts_l1_difference, fit_topics_disagreements, tie_pairs, tie_groups) and entries
(B2..B5 -> eligible, seeds). Counts and topics are recomputed in float64 numpy;
their L1 difference and differing-pair count must each be at most 0.01% of
fit pairs. Centers tied within TIE_RTOL of a pair's maximum Q are ties: a saved
topic disagrees only if it is not tied for the maximum, and counts are compared
summed per tie group (connected components of centers tied in some fit pair).
Runs also record tie_pairs (fit pairs with more than one tied center) and
tie_groups (center groups of size > 1). Each hash is SHA-256
of the saved array's C-order bytes, matching pc_stage1.array_sha256. Both
readings of an entry share its seed list. B1 has no collapse rule. V4's
non-finite held-output check belongs to the scorer; this module reads no held
outputs. For every kept center, post-cut counts must be at least fit_counts;
with all 100 kept, they must be equal. The JSON also records written, in
Europe/Amsterdam local time (YYYY-MM-DD HH:MM). Existing content is compared
without written: identical content is left intact, changed content is refused.
Defaults read and write outputs/percept/ in the main checkout.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE1 = Path('/project/CoSiR/src/test/20261020_claim_test/outputs/percept/paper/stage1')
OUTPUT = STAGE1.parents[1] / 'rcol_stage1.json'
sys.path.insert(0, str(HERE.parent / '20261018_percept_core'))
import pc_fixes

# Saved arrays are float32: Q values within this relative distance of the row maximum
# cannot be told apart, so they are ties under the rule's "lowest index on a tie".
TIE_RTOL = 1e-6


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


def _tie_groups(tied_masks):
    """Connected components (size > 1) of centers tied together in any fit pair's row."""
    parent = list(range(100))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for mask in tied_masks:
        members = np.flatnonzero(mask)
        for other in members[1:]:
            parent[find(int(other))] = find(int(members[0]))
    groups = {}
    for i in range(100):
        groups.setdefault(find(i), []).append(i)
    return sorted(g for g in groups.values() if len(g) > 1)


def _recompute_assignments(centers, fit_latent, masks, saved_topics):
    """Float64 Student-t Q (alpha=1), in blocks to bound memory.

    Returns recomputed counts over all 100 centers, per cut the number of saved
    topics below the row maximum by more than TIE_RTOL (relative), the number of
    fit pairs with more than one tied center, and the distinct tied-center masks.
    """
    centers = centers.astype(np.float64)
    center_squares = np.square(centers).sum(axis=1)
    counts = np.zeros(100, dtype=np.int64)
    disagreements = {cut: 0 for cut in masks}
    tie_pairs = 0
    tied_masks = set()
    for start in range(0, len(fit_latent), 4096):
        latent = fit_latent[start:start + 4096].astype(np.float64)
        distances = (np.square(latent).sum(axis=1)[:, None] + center_squares -
                     2 * (latent @ centers.T))
        np.maximum(distances, 0, out=distances)
        q = 1 / (1 + distances)
        q /= q.sum(axis=1, keepdims=True)
        counts += np.bincount(q.argmax(axis=1), minlength=100)
        tied = q >= q.max(axis=1, keepdims=True) * (1 - TIE_RTOL)
        rows = tied.sum(axis=1) > 1
        tie_pairs += int(rows.sum())
        if rows.any():
            tied_masks.update(map(bytes, np.packbits(np.unique(tied[rows], axis=0), axis=1)))
        for cut, kept in masks.items():
            cut_q = q[:, kept]
            cut_q /= cut_q.sum(axis=1, keepdims=True)
            saved = saved_topics[cut][start:start + len(latent)]
            saved_q = np.take_along_axis(cut_q, saved[:, None], axis=1)[:, 0]
            disagreements[cut] += int(np.count_nonzero(
                saved_q < cut_q.max(axis=1) * (1 - TIE_RTOL)))
    unpacked = [np.unpackbits(np.frombuffer(m, dtype=np.uint8))[:100].astype(bool) for m in tied_masks]
    return counts, disagreements, tie_pairs, unpacked


def read_stage1(path):
    """Read one paper run and return its strict and k40 judgments."""
    path = Path(path)
    version, seed = path.stem.split('__s')
    assert version in ('clip_roberta', 'qwen_qwen') and int(seed) in range(5)
    runs = []
    with np.load(path, allow_pickle=False) as z:
        required = {'centers', 'fit_latent', 'center_norms', 'fit_counts', 'kept_strict', 'kept_k40',
                    'fit_topics_strict', 'fit_topics_k40'}
        assert required <= set(z.files), f'{path}: missing {required - set(z.files)}'
        arrays = {key: z[key] for key in required}
    norms, counts = arrays['center_norms'], arrays['fit_counts']
    centers, latent = arrays['centers'], arrays['fit_latent']
    expected_masks = dict(strict=strict_mask(norms), k40=k40_mask(counts))
    assert counts.sum() == 67327, f'{path}: expected 67,327 fit pairs'
    assert centers.shape == (100, 128) and centers.dtype == np.float32
    assert latent.shape == (67327, 128) and latent.dtype == np.float32
    assert np.isfinite(centers).all() and np.isfinite(latent).all()
    assert np.allclose(norms, np.linalg.norm(centers.astype(np.float64), axis=1), rtol=1e-6), \
        f'{path}: center norms differ from centers'
    judgments = {}
    for cut in ('strict', 'k40'):
        assert arrays[f'fit_topics_{cut}'].shape == (67327,)
        assert np.array_equal(arrays[f'kept_{cut}'], expected_masks[cut]), \
            f'{path}: {cut} mask differs from the rule'
        judgments[cut] = judge(cut, arrays[f'kept_{cut}'], arrays[f'fit_topics_{cut}'])
        kept = arrays[f'kept_{cut}']
        local_counts = np.bincount(arrays[f'fit_topics_{cut}'], minlength=int(kept.sum()))
        # Counts and topics come from separate GPU float32 passes; allow near-ties as below.
        lost = int(np.maximum(counts[kept] - local_counts, 0).sum())
        assert lost <= .0001 * len(latent), f'{path}: {cut} post-cut counts lost {lost} pairs'
        if kept.all():
            moved = int(np.abs(local_counts - counts).sum())
            assert moved <= .0001 * len(latent), f'{path}: {cut} post-cut counts differ by {moved}'
    recomputed_counts, disagreements, tie_pairs, tied_masks = _recompute_assignments(
        centers, latent, expected_masks,
        {cut: arrays[f'fit_topics_{cut}'] for cut in expected_masks})
    # Tied duplicates may split their pairs differently in float32; compare per tie group.
    tie_groups = _tie_groups(tied_masks)
    group_of = np.arange(100)
    for group in tie_groups:
        group_of[group] = group[0]
    count_l1 = int(np.abs(np.bincount(group_of, weights=recomputed_counts, minlength=100) -
                          np.bincount(group_of, weights=counts, minlength=100)).sum())
    assert count_l1 <= .0001 * len(latent), f'{path}: count L1 difference {count_l1}'
    for cut in ('strict', 'k40'):
        assert disagreements[cut] <= .0001 * len(latent), \
            f'{path}: {cut} topic disagreements {disagreements[cut]}'
        keys = ('centers', 'fit_latent', 'center_norms', 'fit_counts', f'kept_{cut}', f'fit_topics_{cut}')
        runs.append(dict(version=version, seed=int(seed), cut=cut, **judgments[cut],
                         artifact=str(path.resolve()), fit_counts_l1_difference=count_l1,
                         fit_topics_disagreements=disagreements[cut],
                         tie_pairs=tie_pairs, tie_groups=tie_groups,
                         array_sha256={k: array_sha256(arrays[k]) for k in keys}))
    return runs


def write_stage1(stage1_dir=STAGE1, output=OUTPUT):
    """Atomically save R-COL once; refuse changed results before held scoring."""
    runs = []
    for version in ('clip_roberta', 'qwen_qwen'):
        for seed in range(5):
            runs.extend(read_stage1(Path(stage1_dir) / f'{version}__s{seed}.npz'))
    report = dict(runs=runs, entries=eligible_entries(runs))
    output = Path(output)
    if output.exists():
        existing = json.loads(output.read_text())
        assert {k: v for k, v in existing.items() if k != 'written'} == report, \
            f'{output}: refusing to overwrite different R-COL content'
        return existing
    report['written'] = datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%d %H:%M')
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
