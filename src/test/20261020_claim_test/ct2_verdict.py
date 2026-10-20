"""Full-scale R-COL and stage 2 verdict, using the frozen stage 1 statistics (CPU)."""
import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import ct_rcol
import ct_score
from ct_score import prepare_held, paired_bootstrap

HERE = Path(__file__).resolve().parent
FULL_ROOT = HERE / 'outputs/stage2/full'
# Frozen stage 1 data stays in the main checkout when running in a worktree.
FROZEN_HELD_LIST = ct_score.OUTPUTS / 'held/held_paintings.json'
FIT_PAIRS = dict(sample=67327, full=216107)


# ct_rcol.read_stage1 has inline 67,327-row gates. Only its count and paths
# are generalized below; masks, judgments and recomputation are reused unchanged.
def read_stage1(path, fit_pairs):
    """Read one paper run and return its strict and k40 judgments."""
    assert fit_pairs > 0
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
    expected_masks = dict(strict=ct_rcol.strict_mask(norms), k40=ct_rcol.k40_mask(counts))
    assert counts.sum() == fit_pairs, f'{path}: expected {fit_pairs:,} fit pairs'
    assert centers.shape == (100, 128) and centers.dtype == np.float32
    assert latent.shape == (fit_pairs, 128) and latent.dtype == np.float32
    assert np.isfinite(centers).all() and np.isfinite(latent).all()
    assert np.allclose(norms, np.linalg.norm(centers.astype(np.float64), axis=1), rtol=1e-6), \
        f'{path}: center norms differ from centers'
    judgments = {}
    for cut in ('strict', 'k40'):
        assert arrays[f'fit_topics_{cut}'].shape == (fit_pairs,)
        assert np.array_equal(arrays[f'kept_{cut}'], expected_masks[cut]), \
            f'{path}: {cut} mask differs from the rule'
        judgments[cut] = ct_rcol.judge(cut, arrays[f'kept_{cut}'], arrays[f'fit_topics_{cut}'])
        kept = arrays[f'kept_{cut}']
        local_counts = np.bincount(arrays[f'fit_topics_{cut}'], minlength=int(kept.sum()))
        # Counts and topics come from separate GPU float32 passes; allow near-ties as below.
        lost = int(np.maximum(counts[kept] - local_counts, 0).sum())
        assert lost <= .0001 * len(latent), f'{path}: {cut} post-cut counts lost {lost} pairs'
        if kept.all():
            moved = int(np.abs(local_counts - counts).sum())
            assert moved <= .0001 * len(latent), f'{path}: {cut} post-cut counts differ by {moved}'
    recomputed_counts, disagreements, tie_pairs, tied_masks = ct_rcol._recompute_assignments(
        centers, latent, expected_masks,
        {cut: arrays[f'fit_topics_{cut}'] for cut in expected_masks})
    # Tied duplicates may split their pairs differently in float32; compare per tie group.
    tie_groups = ct_rcol._tie_groups(tied_masks)
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
                         array_sha256={k: ct_rcol.array_sha256(arrays[k]) for k in keys}))
    return runs


def write_rcol(root, mode='full', stage1_dir=None):
    """Use the original atomic R-COL writer with this mode's fit-pair reader."""
    root = Path(root)
    fit_pairs = FIT_PAIRS[mode]
    stage1_dir = root / 'percept/paper/stage1' if stage1_dir is None else Path(stage1_dir)
    with patch.object(ct_rcol, 'read_stage1', lambda path: read_stage1(path, fit_pairs)):
        return ct_rcol.write_stage1(stage1_dir, root / 'percept/rcol.json')


def stage2_outcome(d, lower):
    """Rule §4 uses the same full-precision boundaries at both scales."""
    return ct_score.stage1_outcome(d, lower).replace('S1-', 'S2-', 1)


def improve_against(result, eligibility):
    """Largest eligible paper entry reaching K1; equal scores use entry ID order."""
    candidates = {entry: value for entry, value in sorted(result['A_e'].items())
                  if entry.split('-')[0] in ct_score.ARM_ORDER[2:]
                  and eligibility['eligible_seeds'].get(entry.split('-')[0])
                  and value >= result['A_K1']}
    return max(candidates, key=candidates.get) if candidates else None


# ct_score.write_verdict embeds stage=1 and S1 outcome IDs; retain its
# serialization, label counting and V4 replacement policy with stage 2 fields.
def write_verdict(path, result, eligibility, held_sha256, labels, code_commit, sources, replace=None):
    """Write the statistics before diagnostics; V1 stage 2a and V2 agreement are required.

    The saved outcome is computed, not licensed for reporting until independent
    V2 reproduction agrees (rule §4). This module computes no diagnostics.
    Changed content needs an explicit V-check rerun reason; record the old hash.
    """
    assert Path(path).name == 'verdict_stage2.json', 'stage 2 verdict path required'
    assert replace is None or replace.strip(), 'replacement needs a non-empty V-check reason'
    dirty_files = subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=all', '--', '.'],
        cwd=HERE, text=True).splitlines()
    counts = {}
    for name, y in labels.items():
        values, ns = np.unique(y[y >= 0], return_counts=True)
        counts[name] = dict(classes={str(v): int(n) for v, n in zip(values, ns)},
                            missing=int((y < 0).sum()))
    verdict = dict(schema=1, stage=2, **result, eligibility=eligibility,
                   held_sha256=held_sha256, code_commit=code_commit, sources=sources,
                   dirty=bool(dirty_files), dirty_files=dirty_files,
                   label_counts=counts, improve_against=improve_against(result, eligibility),
                   outcome_id=stage2_outcome(result['d'], result['interval'][0]),
                   bootstrap=dict(resamples=len(result['bootstrap_d']), rng='numpy.random.default_rng(0)',
                                  arm_order=list(ct_score.ARM_ORDER), percentiles=[2.5, 97.5], method='linear'),
                   reporting_requires='V1 stage 2a pass and independent V2 agreement')
    # JSON makes integer seed keys strings; return exactly the persisted value.
    verdict = json.loads(json.dumps(verdict, allow_nan=False))
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if {k: v for k, v in existing.items() if k != 'replacement'} == verdict:
            return existing
        assert replace is not None, f'{path}: refusing to replace different verdict content without --replace'
        verdict['replacement'] = dict(reason=replace, previous_sha256=ct_score._sha_file(path))
    ct_score._atomic_json(path, verdict)
    return verdict


def run(root=FULL_ROOT, *, stage2a=HERE / 'stage2a.json', device='cuda', workers=8,
        verdict_path=HERE / 'verdict_stage2.json', replace=None):
    """Gate full-scale scoring on stage 2a, freeze R-COL, then save the verdict."""
    from ct2_stage2a import require_pass
    require_pass(stage2a, device)
    import ct_held

    root, stage2a = Path(root), Path(stage2a)
    assert Path(verdict_path).name == 'verdict_stage2.json', 'stage 2 verdict path required'
    stage2a_source = dict(path=str(stage2a.resolve()), sha256=ct_score._sha_file(stage2a))
    write_rcol(root, mode='full')
    rcol_path = root / 'percept/rcol.json'
    rcol, eligible, strict_widths = ct_score.read_rcol(rcol_path)
    rcol_source = dict(path=str(rcol_path.resolve()), sha256=ct_score._sha_file(rcol_path), record=rcol)
    held_list = root / 'held/held_paintings.json'
    names, held_sha256 = ct_held.read_held_list(held_list)
    frozen_sha256 = ct_held.read_held_list(FROZEN_HELD_LIST)[1]
    assert held_sha256 == frozen_sha256, 'stage 2 differs from the stage 1 frozen held list'
    held_list_source = dict(path=str(held_list.resolve()), held_sha256=held_sha256,
                            sha256=ct_score._sha_file(held_list))
    frozen_list_source = dict(path=str(FROZEN_HELD_LIST.resolve()), held_sha256=frozen_sha256,
                              sha256=ct_score._sha_file(FROZEN_HELD_LIST))
    assert names == ct_held.held_names()
    assert ct_held.names_sha256(names) == held_sha256
    labels = ct_held.painting_labels(names)
    assert set(labels) == {'emotion', 'style', 'genre'}
    assert all(y.shape == (ct_score.HELD_ROWS,) and np.issubdtype(y.dtype, np.integer)
               for y in labels.values())
    assert (labels['emotion'] >= 0).any()
    fixed, eligibility, sources = prepare_held(names, held_sha256, eligible, strict_widths,
                                              output_root=root)
    result = paired_bootstrap(fixed, labels['emotion'], eligibility['eligible_seeds'], workers=workers)
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    return write_verdict(verdict_path, result, eligibility, held_sha256, labels, code_commit,
                         dict(held_outputs=sources, rcol=rcol_source, stage2a=stage2a_source,
                              held_list=held_list_source, stage1_held_list=frozen_list_source),
                         replace=replace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=FULL_ROOT,
                        help='Stage 2 artifact root with percept/ and held/ subdirectories')
    parser.add_argument('--mode', choices=FIT_PAIRS, default='full',
                        help='Fit-pair scale; sample is only allowed with --rcol-only')
    parser.add_argument('--rcol-only', action='store_true',
                        help='Freeze <root>/percept/rcol.json before held scoring')
    parser.add_argument('--stage2a', type=Path, default=HERE / 'stage2a.json')
    parser.add_argument('--device', default='cuda', help='Device bound by the stage 2a report')
    parser.add_argument('--verdict', type=Path, default=HERE / 'verdict_stage2.json')
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    parser.add_argument('--replace', metavar='REASON',
                        help='V-check rerun reason for replacing a changed verdict')
    args = parser.parse_args(argv)
    if args.rcol_only:
        write_rcol(args.root, mode=args.mode)
        print(f'Wrote {args.root / "percept/rcol.json"}', flush=True)
        return
    assert args.mode == 'full', 'a stage 2 verdict requires full mode'
    run(args.root, stage2a=args.stage2a, device=args.device, workers=args.workers,
        verdict_path=args.verdict, replace=args.replace)
    print(f'Wrote {args.verdict}; V2 agreement is required before reporting the outcome.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
