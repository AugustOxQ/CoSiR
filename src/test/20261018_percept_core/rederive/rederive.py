#!/root/miniconda3/envs/CoSiR/bin/python
"""Independent C13 scoring. Reads data only; imports no CoSiR modules."""
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_info

ROOT = Path('/tmp/claude-0/-project-CoSiR/482cfe26-f369-448d-99da-12204934cc98/scratchpad/rederive')
SAMPLE = Path('/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz')
HEADS = Path('/project/CoSiR/src/test/20261018_percept_core/outputs/heads')
CLIP = Path('/data/SSD2/pre_extract/artelingo/features')
ROBERTA = Path('/data/SSD2/pre_extract/artelingo_full/roberta_go_emotions')
LABEL_NAMES = ('emotion', 'style', 'genre')
START = time.monotonic()
CHECKS = {}
PROBLEMS = []


def log(message):
    print(f'[{time.monotonic() - START:.1f}s] {message}', flush=True)


def check(name, condition, details):
    CHECKS[name] = {'passed': bool(condition), 'details': details}
    if not condition:
        raise ValueError(f'{name}: {details}')


def distribution(y):
    values, counts = np.unique(y, return_counts=True)
    return {str(int(v)): int(n) for v, n in zip(values, counts)}


def norm_details(x):
    norms = np.linalg.norm(x, axis=1)
    return {'rows': len(x), 'min_norm': float(norms.min()),
            'max_norm': float(norms.max()),
            'max_abs_deviation_from_one': float(np.max(np.abs(norms - 1))),
            'rows_outside_0_01': int(np.count_nonzero(np.abs(norms - 1) > 0.01))}


def normalize(x, name):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    check(name + '_finite_nonzero', np.isfinite(x).all() and np.isfinite(norms).all()
          and (norms > 0).all(), {'shape': list(x.shape), 'dtype': str(x.dtype),
                                  'min_original_norm': float(norms.min()),
                                  'max_original_norm': float(norms.max())})
    x = x / norms[:, None]
    CHECKS[name + '_normalized'] = {'passed': True, 'details': norm_details(x)}
    return x


def join_positions(stored_ids, required_ids, name):
    check(name + '_unique_ids', np.unique(stored_ids).size == stored_ids.size,
          {'stored_rows': int(stored_ids.size),
           'stored_unique_ids': int(np.unique(stored_ids).size),
           'required_rows': int(required_ids.size)})
    order = np.argsort(stored_ids)
    indices = np.searchsorted(stored_ids[order], required_ids)
    in_bounds = indices < len(stored_ids)
    found = np.zeros(len(required_ids), dtype=bool)
    found[in_bounds] = stored_ids[order[indices[in_bounds]]] == required_ids[in_bounds]
    check(name + '_all_required_ids_found', found.all(),
          {'matched': int(found.sum()), 'missing': int((~found).sum()),
           'extra_stored_ids': int(len(stored_ids) - found.sum())})
    positions = order[indices]
    CHECKS[name + '_join_order'] = {'passed': True, 'details': {
        'required_rows_already_at_same_position': int(np.count_nonzero(
            positions == np.arange(len(positions)))), 'join_key': 'sample_id'}}
    return positions


def mean_sd(values):
    a = np.asarray(values, dtype=np.float64)
    return {'mean': float(np.mean(a)), 'sd': float(np.std(a, ddof=0))}


def score(x, labels, name):
    x = np.asarray(x, dtype=np.float32)
    check(name + '_scoring_matrix', x.shape[0] == 5000 and np.isfinite(x).all(),
          {'shape': list(x.shape), 'dtype': str(x.dtype)})
    amis = {label: [] for label in LABEL_NAMES}
    partitions = []
    for seed in range(3):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pred = MiniBatchKMeans(n_clusters=20, random_state=seed, n_init=3,
                                  batch_size=4096).fit_predict(x)
            for label, y in labels.items():
                valid = y >= 0
                amis[label].append(float(adjusted_mutual_info_score(y[valid], pred[valid])))
        warning_text = [str(w.message) for w in caught]
        if warning_text:
            PROBLEMS.extend(f'{name}, k-means seed {seed}: {w}' for w in warning_text)
        partitions.append({'seed': seed, 'cluster_counts': distribution(pred),
                           'warnings': warning_text})
        log(f'{name}: k-means seed {seed} complete')
    return {'shape': list(x.shape), 'partitions': partitions,
            'scores': {label: {**mean_sd(values), 'ami_by_kmeans_seed': values}
                       for label, values in amis.items()}}


def both_sides(img, cap):
    result = {}
    for label in LABEL_NAMES:
        paired = [(a + b) / 2 for a, b in zip(
            img['scores'][label]['ami_by_kmeans_seed'],
            cap['scores'][label]['ami_by_kmeans_seed'])]
        result[label] = {**mean_sd(paired), 'ami_by_kmeans_seed': paired}
        # Definition is explicitly the mean of the two side scores.
        result[label]['mean'] = (img['scores'][label]['mean'] +
                                 cap['scores'][label]['mean']) / 2
    return result


def painting_means(captions, pid, counts, subset, name):
    normalized = normalize(captions, name + '_captions')
    sums = np.zeros((len(counts), normalized.shape[1]), dtype=np.float32)
    np.add.at(sums, pid, normalized)
    means = sums / counts.astype(np.float32)[:, None]
    means = normalize(means, name + '_painting_means')
    return np.ascontiguousarray(means[subset], dtype=np.float32)


def main():
    log('Begin independent re-derivation')
    with np.load(SAMPLE, allow_pickle=False) as archive:
        sample = {key: archive[key] for key in archive.files}
    ids, pid, first, subset = (sample[k] for k in ('sample_ids', 'pid', 'first', 'subset_pid'))
    n, paintings = len(ids), len(first)
    check('sample_counts', n == 99999 and paintings == 19889 and len(subset) == 5000,
          {'captions': n, 'paintings': paintings, 'evaluation_paintings': len(subset)})
    check('sample_id_dtype_and_uniqueness', ids.dtype == np.int64 and len(np.unique(ids)) == n,
          {'dtype': str(ids.dtype), 'unique_sample_ids': len(np.unique(ids))})
    check('pid_range', pid.min() == 0 and pid.max() == paintings - 1,
          {'min': int(pid.min()), 'max': int(pid.max())})
    counts = np.bincount(pid, minlength=paintings)
    check('all_paintings_have_captions', (counts > 0).all(),
          {'min_captions': int(counts.min()), 'max_captions': int(counts.max()),
           'caption_count_distribution': distribution(counts), 'sum': int(counts.sum())})
    actual_first = np.full(paintings, n, dtype=np.int64)
    np.minimum.at(actual_first, pid, np.arange(n))
    check('first_caption_indices', np.array_equal(first, actual_first),
          {'checked_paintings': paintings})
    check('painting_names_by_pid', np.array_equal(sample['painting_ids'], sample['all_painting_ids'][pid]),
          {'checked_captions': n})
    check('unique_painting_names', len(np.unique(sample['all_painting_ids'])) == paintings,
          {'unique_names': len(np.unique(sample['all_painting_ids']))})
    check('subset_valid_unique', len(np.unique(subset)) == 5000 and subset.min() >= 0
          and subset.max() < paintings, {'unique_pids': len(np.unique(subset))})
    check('subset_name_order', np.array_equal(sample['subset_painting_ids'], sample['all_painting_ids'][subset]),
          {'checked_paintings': len(subset)})
    CHECKS['auxiliary_subset_image_indices'] = {'passed': True, 'details': {
        'used_for_scoring': False,
        'matches_first_caption_row': int(np.count_nonzero(sample['subset_image_indices'] == first[subset])),
        'note': 'Field semantics are not specified by the task; use first[subset_pid] for image sample IDs.'}}
    emo = sample['emo']
    check('emotion_class_range', ((emo >= -1) & (emo <= 8)).all(), distribution(emo))
    votes = np.zeros((paintings, 9), dtype=np.int64)
    valid = emo >= 0
    np.add.at(votes, (pid[valid], emo[valid]), 1)
    majority = np.argmax(votes, axis=1)  # First maximum implements smallest-id tie breaking.
    no_votes = votes.sum(axis=1) == 0
    majority[no_votes] = -1
    tied = (votes == votes.max(axis=1)[:, None]).sum(axis=1) > 1
    CHECKS['emotion_majority'] = {'passed': True, 'details': {
        'tie_rule': 'smallest class id',
        'all_paintings_with_tied_nonempty_majority': int((tied & ~no_votes).sum()),
        'evaluation_paintings_with_tied_nonempty_majority': int((tied & ~no_votes)[subset].sum()),
        'all_paintings_without_emotion': int(no_votes.sum())}}
    labels = {'emotion': majority[subset], 'style': sample['sty'][first[subset]],
              'genre': sample['gen'][first[subset]]}
    label_counts = {label: {'valid': int((y >= 0).sum()), 'excluded': int((y < 0).sum()),
                            'class_counts_including_missing': distribution(y)}
                    for label, y in labels.items()}
    CHECKS['evaluation_label_counts'] = {'passed': True, 'details': label_counts}
    log('Sample structure and labels checked: ' + str(label_counts))

    specifications = [('row 6', 'Ddef', False, LABEL_NAMES, range(3)),
                      ('row 10', 'R0o', False, ('emotion', 'style'), range(3)),
                      ('row 13', 'R0o', True, LABEL_NAMES, range(3)),
                      ('gate', 'R0', False, ('genre',), (0,))]
    expected = {}
    for group, graph, prompted, aspects, seeds in specifications:
        for aspect in aspects:
            for seed in seeds:
                filename = f'{aspect}__{graph}' + ('__prompted' if prompted else '') + f'__s{seed}.npz'
                expected[filename] = {'group': group, 'graph': graph, 'prompted': prompted,
                                      'aspect': aspect, 'seed': seed}
    found = {p.name for p in HEADS.glob('*.npz') if not p.name.endswith('.pending.npz')}
    pending = sorted(p.name for p in HEADS.glob('*.pending.npz'))
    inventory = {'expected': len(expected), 'found': len(found),
                 'missing': sorted(set(expected) - found), 'unexpected': sorted(found - set(expected)),
                 'pending_skipped': pending}
    CHECKS['head_inventory'] = {'passed': not inventory['missing'] and not inventory['unexpected'],
                                 'details': inventory}
    if inventory['missing'] or inventory['unexpected']:
        PROBLEMS.append('Head inventory mismatch: ' + str(inventory))
    if inventory['missing']:
        raise ValueError('Missing required head files')
    head_norms = {}
    largest_deviation = 0.0
    for filename in sorted(found):
        with np.load(HEADS / filename, allow_pickle=False) as z:
            check(filename + '_arrays', set(z.files) == {'img', 'cap'}, {'array_names': z.files})
            head_norms[filename] = {}
            for side in ('img', 'cap'):
                stored = z[side]
                check(filename + '_' + side + '_stored_shape_dtype',
                      stored.shape == (5000, 2048) and stored.dtype == np.float16,
                      {'shape': list(stored.shape), 'dtype': str(stored.dtype)})
                x = stored.astype(np.float32)
                info = norm_details(x)
                check(filename + '_' + side + '_unit_norm', np.isfinite(x).all()
                      and info['rows_outside_0_01'] == 0, info)
                head_norms[filename][side] = info
                largest_deviation = max(largest_deviation, info['max_abs_deviation_from_one'])
    log(f'All heads checked before scoring; largest norm deviation: {largest_deviation!r}')

    # Read only the permitted FeatureManager data, directly from its NumPy shards.
    with open(CLIP / 'metadata.json') as handle:
        metadata = json.load(handle)
    shard_dirs = sorted(p for p in (CLIP / 'shards').iterdir() if p.is_dir())
    shard_ids = [np.load(p / 'sample_ids.npy', allow_pickle=False) for p in shard_dirs]
    stored_ids = np.concatenate(shard_ids)
    check('clip_metadata_counts', len(stored_ids) == metadata['total_samples']
          and len(shard_dirs) == metadata['num_shards'],
          {'metadata': metadata, 'observed_rows': len(stored_ids), 'observed_shards': len(shard_dirs)})
    positions = join_positions(stored_ids, ids, 'clip_caption_join')
    clip_captions = np.empty((n, 512), dtype=np.float32)
    clip_images = np.empty((5000, 512), dtype=np.float32)
    image_positions = positions[first[subset]]
    check('clip_image_id_join', np.array_equal(stored_ids[image_positions], ids[first[subset]]),
          {'matched': 5000, 'requested_sample_ids': 'sample_ids[first[subset_pid]]'})
    offset = 0
    for shard, these_ids in zip(shard_dirs, shard_ids):
        stop = offset + len(these_ids)
        for feature, destination, wanted in (
                ('txt_features', clip_captions, positions),
                ('img_features', clip_images, image_positions)):
            vectors = np.load(shard / (feature + '.npy'), mmap_mode='r', allow_pickle=False)
            check(shard.name + '_' + feature + '_shape_dtype',
                  vectors.shape == (len(these_ids), 512) and vectors.dtype == np.float32,
                  {'shape': list(vectors.shape), 'dtype': str(vectors.dtype)})
            selected = (wanted >= offset) & (wanted < stop)
            destination[selected] = vectors[wanted[selected] - offset]
        offset = stop
    image_x = normalize(clip_images, 'clip_evaluation_images')
    clip_cap_x = painting_means(clip_captions, pid, counts, subset, 'clip')
    with np.load(ROBERTA / 'ids.npz', allow_pickle=False) as z:
        roberta_ids, roberta_names = z['sample_ids'], z['painting_ids']
    roberta_vectors = np.load(ROBERTA / 'txt.npy', mmap_mode='r', allow_pickle=False)
    check('roberta_shape', roberta_vectors.shape == (99999, 768)
          and roberta_ids.shape == (99999,) and roberta_names.shape == (99999,),
          {'vectors_shape': list(roberta_vectors.shape), 'vectors_dtype': str(roberta_vectors.dtype),
           'id_shape': list(roberta_ids.shape), 'painting_id_shape': list(roberta_names.shape)})
    roberta_positions = join_positions(roberta_ids, ids, 'roberta_caption_join')
    name_matches = roberta_names[roberta_positions] == sample['painting_ids']
    check('roberta_painting_ids_agree', name_matches.all(),
          {'matched': int(name_matches.sum()), 'mismatched': int((~name_matches).sum())})
    roberta_cap_x = painting_means(roberta_vectors[roberta_positions], pid, counts, subset, 'roberta')
    log('All feature joins, caption means, and normalization checks complete')

    runs = []
    for filename, spec in expected.items():
        with np.load(HEADS / filename, allow_pickle=False) as z:
            img = score(z['img'].astype(np.float32), labels, filename + '/img')
            cap = score(z['cap'].astype(np.float32), labels, filename + '/cap')
        both = both_sides(img, cap)
        runs.append({**spec, 'file': filename, 'img': img, 'cap': cap,
                     'both_sides': both, 'target_value': both[spec['aspect']]['mean']})
    aggregates = []
    for group, _, _, aspects, seeds in specifications:
        for aspect in aspects:
            relevant = sorted((r for r in runs if r['group'] == group and r['aspect'] == aspect),
                              key=lambda r: r['seed'])
            sides = {}
            for side in ('img', 'cap', 'both_sides'):
                sides[side] = {}
                for label in LABEL_NAMES:
                    values = [(r[side]['scores'] if side != 'both_sides' else r[side])[label]['mean']
                              for r in relevant]
                    sides[side][label] = {**mean_sd(values), 'values_by_run_seed': values}
            targets = [r['target_value'] for r in relevant]
            aggregates.append({'group': group, 'aspect': aspect,
                               'run_seeds': [r['seed'] for r in relevant], 'sides': sides,
                               'target': {**mean_sd(targets), 'values_by_run_seed': targets}})

    b_img = score(image_x, labels, 'row1/clip_img')
    b_clip = score(clip_cap_x, labels, 'row1/clip_cap')
    b_roberta = score(roberta_cap_x, labels, 'row1/roberta_cap')
    part_b = {'img_clip': b_img, 'cap_clip': b_clip, 'cap_roberta': b_roberta,
              'both_clip': both_sides(b_img, b_clip),
              'both_roberta': both_sides(b_img, b_roberta)}
    result = {
        'rule': 'C13; independent re-derivation from stored data',
        'runtime': {'python': sys.version, 'executable': sys.executable,
                    'numpy': np.__version__, 'sklearn': sklearn.__version__,
                    'scipy': scipy.__version__, 'threadpools': threadpool_info(),
                    'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS'],
                    'MKL_NUM_THREADS': os.environ['MKL_NUM_THREADS'], 'device': 'CPU',
                    'elapsed_seconds': time.monotonic() - START},
        'method': {'kmeans': {'n_clusters': 20, 'random_states': [0, 1, 2], 'n_init': 3,
                              'batch_size': 4096, 'other_parameters': 'sklearn defaults'},
                   'ami': 'adjusted_mutual_info_score, default arguments, y >= 0',
                   'sd_ddof': 0, 'head_normalization': 'none; cast stored float16 to float32',
                   'caption_mean': 'float32 normalized caption sums in sample order / caption count; normalize each mean',
                   'group_sd': 'population SD of the per-run three-kmeans-seed mean scores',
                   'both_sides_per_seed': 'paired kmeans seeds, average of img and cap AMIs'},
        'inputs': {'sample': str(SAMPLE), 'heads': str(HEADS), 'clip': str(CLIP),
                   'roberta': str(ROBERTA)},
        'label_counts': label_counts,
        'evaluation_labels': {k: v.tolist() for k, v in labels.items()},
        'head_inventory': inventory, 'head_norms': head_norms,
        'largest_head_norm_deviation': largest_deviation,
        'part_a': {'runs': runs, 'group_aspect_aggregates': aggregates},
        'part_b': part_b, 'checks': CHECKS, 'problems': PROBLEMS,
        'deletions': {'performed': [], 'would_remove': []}}
    output = ROOT / 'rederive_results.json'
    with open(output, 'w') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    # Render only from the persisted full-precision result file.
    with open(output) as handle:
        saved = json.load(handle)
    write_summary(saved)
    log(f'DONE: {len(runs)} head runs, {len(runs) * 2 + 3} matrices, '
        f'{(len(runs) * 2 + 3) * 3} k-means fits; problems={len(PROBLEMS)}')


def write_summary(saved):
    lines = ['# Independent C13 re-derivation', '',
             'Part A values are means and population SDs over run seeds. Each side score first '
             'averages AMI over k-means seeds 0, 1, 2. Target rows have label equal to aspect; '
             'the both-sides target aggregate is included in those rows and explicitly in JSON.', '',
             '| Group | Aspect | Side | Label | Target label | Mean | SD over run seeds |',
             '|---|---|---|---|---|---:|---:|']
    for group in saved['part_a']['group_aspect_aggregates']:
        if group['group'] == 'gate':
            continue
        for side, scores in group['sides'].items():
            for label, value in scores.items():
                lines.append(f"| {group['group']} | {group['aspect']} | {side} | {label} | "
                             f"{'yes' if label == group['aspect'] else ''} | "
                             f"{value['mean']:.6f} | {value['sd']:.6f} |")
    lines += ['', '## Gate: genre / R0 / default / run seed 0', '',
              'SD below is over the three k-means seeds. Both-sides SD uses paired img/cap seeds. '
              'With one run seed, the SD over run seeds is zero (recorded in JSON).', '',
              '| Side | Label | Mean | SD over k-means seeds |', '|---|---|---:|---:|']
    gate = next(r for r in saved['part_a']['runs'] if r['group'] == 'gate')
    for side in ('img', 'cap', 'both_sides'):
        scores = gate[side]['scores'] if side != 'both_sides' else gate[side]
        for label, value in scores.items():
            lines.append(f"| {side} | {label} | {value['mean']:.6f} | {value['sd']:.6f} |")
    lines += ['', '## Part B: September encoders', '',
              'SD is over the three k-means seeds; both-sides SD uses paired img/cap seeds.', '',
              '| Side / encoder | Label | Mean | SD over k-means seeds |', '|---|---|---:|---:|']
    for side, value in saved['part_b'].items():
        scores = value['scores'] if 'scores' in value else value
        for label, score_value in scores.items():
            lines.append(f"| {side} | {label} | {score_value['mean']:.6f} | {score_value['sd']:.6f} |")
    lines += ['', '## Checks', '',
              f"Largest head-row norm deviation after float32 cast, before any scoring: "
              f"{saved['largest_head_norm_deviation']!r}; required tolerance: 0.01.", '',
              'Each check and its observed counts are listed below. Class-count key -1 denotes excluded labels. '
              'The JSON also contains all 5,000 evaluation labels, per-file norm statistics, '
              'per-fit cluster counts, and all individual k-means-seed AMIs.', '']
    for name, value in saved['checks'].items():
        lines.append(f"- **{'PASS' if value['passed'] else 'FAIL'} {name}**: "
                     + json.dumps(value['details'], ensure_ascii=False, sort_keys=True))
    lines += ['', '## Problems and file handling', '']
    lines += ['- ' + problem for problem in saved['problems']] if saved['problems'] else ['- No problems detected.']
    lines += ['- No files deleted. Would remove: none.',
              '- No project source code, prior results, reports, or unlisted input files were read.',
              '- Computation used CPU only, with OMP_NUM_THREADS=8 and MKL_NUM_THREADS=8.',
              f"- Versions: NumPy {saved['runtime']['numpy']}, scikit-learn {saved['runtime']['sklearn']}, "
              f"SciPy {saved['runtime']['scipy']}.", '']
    with open(ROOT / 'rederive_summary.md', 'w') as handle:
        handle.write('\n'.join(lines))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        log('FAILED')
        traceback.print_exc()
        raise
