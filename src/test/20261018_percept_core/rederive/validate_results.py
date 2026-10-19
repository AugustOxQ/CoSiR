#!/root/miniconda3/envs/CoSiR/bin/python
"""Audit the persisted independent results and their rendered tables."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/tmp/claude-0/-project-CoSiR/482cfe26-f369-448d-99da-12204934cc98/scratchpad/rederive')


def same(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


def verify_scores(scores):
    assert set(scores) == {'emotion', 'style', 'genre'}
    for value in scores.values():
        seeds = value['ami_by_kmeans_seed']
        assert len(seeds) == 3 and np.isfinite(seeds).all()
        same(value['mean'], np.mean(seeds))
        same(value['sd'], np.std(seeds, ddof=0))


def main():
    with open(ROOT / 'rederive_results.json') as handle:
        result = json.load(handle)
    summary = (ROOT / 'rederive_summary.md').read_text()
    assert all(check['passed'] for check in result['checks'].values())
    assert not result['problems']
    assert result['largest_head_norm_deviation'] <= 0.01
    assert not result['head_inventory']['missing']
    assert not result['head_inventory']['unexpected']
    assert len(result['head_norms']) == 25
    expected_files = set()
    for group, graph, prompted, aspects in (
            ('row 6', 'Ddef', False, ('emotion', 'style', 'genre')),
            ('row 10', 'R0o', False, ('emotion', 'style')),
            ('row 13', 'R0o', True, ('emotion', 'style', 'genre')),
            ('gate', 'R0', False, ('genre',))):
        for aspect in aspects:
            for seed in ((0,) if group == 'gate' else range(3)):
                expected_files.add(f'{aspect}__{graph}' + ('__prompted' if prompted else '') + f'__s{seed}.npz')
    runs = result['part_a']['runs']
    assert len(runs) == 25 and {r['file'] for r in runs} == expected_files
    matrices = []
    for run in runs:
        for side in ('img', 'cap'):
            matrix = run[side]
            matrices.append(matrix)
            assert matrix['shape'] == [5000, 2048]
            verify_scores(matrix['scores'])
        verify_scores(run['both_sides'])
        for label, value in run['both_sides'].items():
            expected = (np.array(run['img']['scores'][label]['ami_by_kmeans_seed']) +
                        np.array(run['cap']['scores'][label]['ami_by_kmeans_seed'])) / 2
            same(value['ami_by_kmeans_seed'], expected)
        same(run['target_value'], run['both_sides'][run['aspect']]['mean'])
    groups = result['part_a']['group_aspect_aggregates']
    assert len(groups) == 9
    for group in groups:
        relevant = sorted([r for r in runs if r['group'] == group['group'] and r['aspect'] == group['aspect']],
                          key=lambda r: r['seed'])
        assert group['run_seeds'] == [r['seed'] for r in relevant]
        for side, scores in group['sides'].items():
            for label, value in scores.items():
                numbers = [(r[side]['scores'] if side != 'both_sides' else r[side])[label]['mean']
                           for r in relevant]
                same(value['values_by_run_seed'], numbers)
                same(value['mean'], np.mean(numbers))
                same(value['sd'], np.std(numbers, ddof=0))
                if group['group'] != 'gate':
                    row = f"| {group['group']} | {group['aspect']} | {side} | {label} | "
                    row += f"{'yes' if label == group['aspect'] else ''} | {value['mean']:.6f} | {value['sd']:.6f} |"
                    assert row in summary
        same(group['target']['mean'], group['sides']['both_sides'][group['aspect']]['mean'])
        same(group['target']['sd'], group['sides']['both_sides'][group['aspect']]['sd'])
    gate = next(r for r in runs if r['group'] == 'gate')
    for side in ('img', 'cap', 'both_sides'):
        scores = gate[side]['scores'] if side != 'both_sides' else gate[side]
        for label, value in scores.items():
            assert f"| {side} | {label} | {value['mean']:.6f} | {value['sd']:.6f} |" in summary
    b = result['part_b']
    for side, value in b.items():
        scores = value['scores'] if 'scores' in value else value
        for label, value in scores.items():
            assert f"| {side} | {label} | {value['mean']:.6f} | {value['sd']:.6f} |" in summary
    for side in ('img_clip', 'cap_clip', 'cap_roberta'):
        matrices.append(b[side])
        verify_scores(b[side]['scores'])
    for name, cap in (('both_clip', 'cap_clip'), ('both_roberta', 'cap_roberta')):
        verify_scores(b[name])
        for label, value in b[name].items():
            expected = (np.array(b['img_clip']['scores'][label]['ami_by_kmeans_seed']) +
                        np.array(b[cap]['scores'][label]['ami_by_kmeans_seed'])) / 2
            same(value['ami_by_kmeans_seed'], expected)
    for matrix in matrices:
        assert [p['seed'] for p in matrix['partitions']] == [0, 1, 2]
        for partition in matrix['partitions']:
            assert sum(partition['cluster_counts'].values()) == 5000
            assert len(partition['cluster_counts']) <= 20
            assert not partition['warnings']
    assert len(matrices) == 53
    for label, y in result['evaluation_labels'].items():
        y = np.asarray(y)
        counts = result['label_counts'][label]
        assert len(y) == 5000
        assert int((y >= 0).sum()) == counts['valid']
        assert int((y < 0).sum()) == counts['excluded']
    # 72 Part A rows + 9 gate rows + 15 Part B rows; three headers are excluded.
    all_table_rows = [line for line in summary.splitlines() if line.startswith('| ')]
    assert len(all_table_rows) == 99
    print('PASS: 25 head runs, 9 group/aspect aggregates, 53 matrices, 159 k-means partitions.')
    print('PASS: 477 original AMIs; all side/seed/group means and population SDs, paired both-sides values, targets.')
    print('PASS: label counts, head norms, feature-join checks, and 96 summary data rows.')
    print('Largest head norm deviation:', repr(result['largest_head_norm_deviation']))
    print('Label counts:', json.dumps(result['label_counts'], sort_keys=True))
    print('Problems:', result['problems'])
    audit_note = ('\n## Saved-output audit\n\n'
                  '- validate_results.py verified all 477 original AMIs, all reported means and population SDs, '
                  'paired both-sides values, target aggregates, label counts, 159 partition counts, '
                  'and the values and formatting of all 96 summary table rows.\n')
    if '## Saved-output audit' not in summary:
        with open(ROOT / 'rederive_summary.md', 'a') as handle:
            handle.write(audit_note)


if __name__ == '__main__':
    main()
