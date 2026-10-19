"""CPU PercepT image-side scores, frozen baselines and comparator sheet."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
for folder in ('20261017_image_transfer_pilot', '20261018_purified_buddy_pilot'):
    sys.path.insert(0, str(HERE.parent / folder))
from itp_data import atomic_json, default_images, fingerprint, load_frozen, targets
from pbp_graphs import label_vectors, sha_file
from pav_score import ami, partitions, summary
from apc_data import l2, paint_mean
from pc_fixes import top_used, unit_tests_pass

OUTPUTS = HERE / 'outputs'
LABELS = ('emotion', 'style', 'genre')
ROW2 = dict(emotion=0.08855639551158387, style=0.17095358715401665,
            genre=0.3930761231906966)
ARMS = dict(P1='clip_roberta', P2='clip_roberta', P3='qwen_qwen',
            P4='qwen_qwen', P5='qwen_roberta')


def ami_scores(x, labels):
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2 and x.shape[0] == 5000 and x.shape[1] > 0
    assert np.isfinite(x).all()
    assert np.max(np.abs(np.linalg.norm(x, axis=1) - 1)) < .01
    assert all(y.shape == (5000,) for y in labels.values())
    ls = partitions(x, 20)
    return {label: summary([ami(y[y >= 0], lab[y >= 0]) for lab in ls])
            for label, y in labels.items()}


def both_mean(img, cap):
    return {label: (img[label]['mean'] + cap[label]['mean']) / 2 for label in img}


def native_ami(prob, labels):
    prob = np.asarray(prob)
    assert prob.ndim == 2 and prob.shape[1] > 0 and np.isfinite(prob).all()
    assert ((prob >= 0) & (prob <= 1)).all()
    assert all(y.shape == (len(prob),) for y in labels.values())
    topic = prob.argmax(axis=1)
    return {label: float(ami(y[y >= 0], topic[y >= 0])) for label, y in labels.items()}


def aggregate_arm(seeds, stage0_pass, tests_pass, fix_tests=None, counts=None):
    assert set(seeds) <= {'0', '1', '2'}
    reasons = []
    if not stage0_pass:
        reasons.append('Stage 0 did not PASS')
    if not tests_pass:
        reasons.append('Unit tests (spec §6) not recorded' if fix_tests is None
                       else f'Unit tests (spec §6) failed: {fix_tests}')
    missing = sorted({'0', '1', '2'} - set(seeds))
    if missing:
        reasons.append('Incomplete Stage 1/2 files for seeds ' + ', '.join(missing))
    for seed, result in seeds.items():
        assert result['K'] == 40 and 'reason' not in result
        auc = result['dev_auc']
        assert auc is None or (np.isfinite(auc) and 0 <= auc <= 1)
        if auc is None:
            reasons.append(f'Seed {seed} dev AUC undefined')
        elif auc < .75:
            reasons.append(f'Seed {seed} dev AUC below 0.75')
    result = dict(established=not reasons, reasons=reasons, n_seeds=len(seeds),
                  img=None, native_img=None, native_topic=None, K=None, K_gamma=None,
                  kept_topics=None, dev_auc=None)
    counts = list(seeds.values()) if counts is None else counts
    if counts:
        assert all(r['K'] == 40 for r in counts)
        result['K'] = summary([r['K'] for r in counts])
        result['K_gamma'] = {g: summary([r['K_gamma'][g] for r in counts])
                             for g in ('1.5', '2', '3', '5')}
        result['kept_topics'] = {key: summary([r['kept_topics'][key] for r in counts])
                                 for key in ('fit_pairs', 'with_fit_pair', 'with_1pct', 'share_before_cut')}
    if missing:
        return result
    result['img'] = {a: summary([r['img'][a]['mean'] for r in seeds.values()]) for a in LABELS}
    for key in ('native_img', 'native_topic'):
        result[key] = {a: summary([r[key][a] for r in seeds.values()]) for a in LABELS}
    if all(r['dev_auc'] is not None for r in seeds.values()):
        result['dev_auc'] = summary([r['dev_auc'] for r in seeds.values()])
    return result


def comparators(arms):
    result = {}
    for aspect in LABELS:
        result[aspect] = {}
        for name, established_only in (('established', True), ('best_all', False)):
            eligible = {a: r for a, r in arms.items()
                        if r['img'] is not None and (r['established'] or not established_only)}
            if not eligible:
                result[aspect][name] = None
                continue
            highest_arm = min(eligible, key=lambda a: (-eligible[a]['img'][aspect]['mean'], int(a[1:])))
            highest = eligible[highest_arm]['img'][aspect]['mean']
            tied = [a for a, r in eligible.items() if highest - r['img'][aspect]['mean'] <= .003 + 1e-12]
            arm = min(tied, key=lambda a: int(a[1:]))
            result[aspect][name] = dict(arm=arm, mean=eligible[arm]['img'][aspect]['mean'],
                                        highest_arm=highest_arm, highest_mean=highest, established=eligible[arm]['established'])
    return result


def row1(s, labels):
    from pc_data import caption_features, image_features
    subset = s['subset_pid']
    img = ami_scores(l2(np.asarray(image_features(s, 'clip_image'), dtype=np.float32))[subset], labels)
    result = {}
    for kind in ('clip_text', 'roberta'):
        captions = l2(np.asarray(caption_features(s, kind), dtype=np.float32))
        cap = ami_scores(l2(paint_mean(captions, s['pid'], len(s['first'])))[subset], labels)
        result[kind] = dict(img=img, cap=cap, both_mean=both_mean(img, cap))
    return result


def score_seed(stage1, stage2, s, labels):
    meta1 = json.loads(stage1.with_suffix('.json').read_text())
    meta2 = json.loads(stage2.with_suffix('.json').read_text())
    assert meta1['sample_sha256'] == fingerprint(s), 'Stage 1 sample mismatch'
    hashes = dict(stage1=sha_file(stage1))
    assert meta2['stage1_sha256'] == hashes['stage1'], 'Stage 2 used a different Stage 1 file'
    counts = meta1['K_gamma']
    assert set(counts) == {'1.5', '2', '3', '5'}
    assert all(isinstance(k, int) and 1 <= k <= 100 for k in counts.values())
    k = meta1['K']
    assert type(k) is int and k == 40 == meta2['K']
    assert 'reason' not in meta2, f'{stage2.name}: unexpected reason'
    hashes['stage2'] = sha_file(stage2)
    with np.load(stage1, allow_pickle=False) as z:
        assert set(z.files) == {'center_norms', 'fit_counts', 'kept', 'topic', 'q_paint_eval'}
        norms, kept, topic, q = (z[key] for key in ('center_norms', 'kept', 'topic', 'q_paint_eval'))
        fit_counts = z['fit_counts']
    assert fit_counts.shape == (100,) and fit_counts.dtype == np.int64 and (fit_counts >= 0).all()
    assert norms.shape == (100,) and norms.dtype == np.float32 and np.isfinite(norms).all()
    assert kept.shape == (100,) and kept.dtype == bool and k == 40 == kept.sum()
    assert np.array_equal(kept, top_used(fit_counts, 40)), 'Stage 1 kept mask mismatch'
    assert topic.shape == (99999,) and topic.dtype == np.int16
    assert ((topic >= 0) & (topic < k)).all()
    assert q.shape == (5000, k) and q.dtype == np.float32
    assert np.allclose(q.sum(axis=1), 1, atol=1e-5)
    with np.load(stage2, allow_pickle=False) as z:
        prob = z['eval_prob']
    assert prob.shape == (5000, k) and prob.dtype == np.float32
    assert 0 <= meta2['dev_valid_topics'] <= k
    return dict(img=ami_scores(l2(prob), labels), native_img=native_ami(prob, labels),
                native_topic=native_ami(q, labels), K=k, K_gamma=counts,
                kept_topics=meta1['kept_topics'], dev_auc=meta2['dev_auc'],
                dev_valid_topics=meta2['dev_valid_topics'], source_sha256=hashes,
                config=dict(stage1=meta1['config'], stage2=meta2['config']))


def score_outputs(s, labels, outdir=OUTPUTS):
    outdir = Path(outdir)
    stage0_path = outdir / 'stage0.json'
    stage0 = json.loads(stage0_path.read_text()) if stage0_path.exists() else dict(status='MISSING')
    repeat_path = outdir / 'stage1_repeat.json'
    repeat = dict(status='missing', bitwise_equal=None, arrays=None, sidecar=None)
    if repeat_path.exists():
        report = json.loads(repeat_path.read_text())
        first = outdir / 'stage1/clip_roberta__s0.npz'
        repeat['status'] = 'stale'
        if first.exists() and report.get('first_npz_sha256') == sha_file(first):
            repeat = dict(status='fresh', **{key: report[key]
                          for key in ('bitwise_equal', 'arrays', 'sidecar')})
    fix_tests = stage0.get('fix_tests')
    tests_pass = unit_tests_pass(fix_tests)
    arms, missing = {}, {}
    for arm, version in ARMS.items():
        seeds = {}
        stage1_seeds = {}
        for seed in (0, 1, 2):
            stage1 = outdir / 'stage1' / f'{version}__s{seed}.npz'
            stage2 = outdir / 'stage2' / f'{arm}__s{seed}.npz'
            if stage1.exists() and stage1.with_suffix('.json').exists():
                meta1 = json.loads(stage1.with_suffix('.json').read_text())
                assert meta1['sample_sha256'] == fingerprint(s), 'Stage 1 sample mismatch'
                assert type(meta1['K']) is int and meta1['K'] == 40
                assert set(meta1['K_gamma']) == {'1.5', '2', '3', '5'}
                assert all(isinstance(k, int) and 1 <= k <= 100 for k in meta1['K_gamma'].values())
                stage1_seeds[str(seed)] = {key: meta1[key] for key in ('K', 'K_gamma', 'kept_topics')}
            paths = (stage1, stage1.with_suffix('.json'), stage2, stage2.with_suffix('.json'))
            absent = [str(p.relative_to(outdir)) for p in paths if not p.exists()]
            if absent:
                missing[f'{arm}__s{seed}'] = absent
                continue
            seeds[str(seed)] = score_seed(stage1, stage2, s, labels)
        arms[arm] = dict(version=version, seeds=seeds, stage1_seeds=stage1_seeds,
                         **aggregate_arm(seeds, stage0['status'] == 'PASS', tests_pass, fix_tests,
                                         list(stage1_seeds.values())))
    return dict(schema=2, sample_sha256=fingerprint(s), stage0=stage0, stage1_repeat=repeat, arms=arms,
                comparator=comparators(arms), missing_outputs=missing,
                conventions='Image only: l2(eval_prob), pav k20 seeds012, y>=0; '
                'population sd over PercepT seeds012; no PercepT caption/mean. '
                'D15: within 0.003 of highest mean, lowest arm number; same rule for best_all. '
                'Only complete three-seed arms enter comparator selection.')


def check_row2(s=None, outdir=OUTPUTS):
    s = load_frozen() if s is None else s
    labels = label_vectors(s, s['subset_pid'])
    img = ami_scores(default_images(s)[0][s['subset_pid']], labels)
    cap = ami_scores(targets(s, 'default')[s['subset_pid']], labels)
    means = both_mean(img, cap)
    checks = {label: abs(means[label] - ROW2[label]) <= 1e-6 for label in LABELS}
    result = dict(status='PASS' if all(checks.values()) else 'FAIL',
                  img=img, cap=cap, both_mean=means, expected=ROW2, checks=checks,
                  sample_sha256=fingerprint(s))
    atomic_json(Path(outdir) / 'row2_check.json', result)
    return result


def markdown(result):
    fmt = lambda v: f"{v['mean']:.5f} ± {v['sd']:.5f}" if v is not None else '—'
    repeat = result.get('stage1_repeat', dict(status='missing', bitwise_equal=None))
    equal = '—' if repeat['bitwise_equal'] is None else 'yes' if repeat['bitwise_equal'] else 'no'
    repeat_line = f"Stage 1 repeat (A2, reported, not a gate): {repeat['status']}; bitwise equal: {equal}"
    for kind in ('arrays', 'sidecar'):
        differences = sorted(key for key, same in (repeat.get(kind) or {}).items() if not same)
        if differences:
            repeat_line += f"; {kind}: {', '.join(differences)}"
    lines = ['# PercepT core results', '',
             'Development evaluation on 5,000 frozen paintings; selection uses this same sample. '
             'AMI masks missing labels. Spreads are population SD; no significance tests.', '',
             'PercepT has an image-only path; its caption and both-side mean columns are empty. '
             'Arm spreads use the three PercepT seeds, after averaging three k-means seeds. '
             'Row 1 spreads use the three k-means seeds.', '',
             f"Stage 0: {result['stage0']['status']}. Established requires Stage 0 PASS, the unit tests "
             'passing (spec section 6 (b), line below), complete Stage 1/2 files and dev AUC >= 0.75 on every seed.', '',
             repeat_line, '',
             'Unit tests (spec section 6, `pytest -q test_pc_fixes.py`): '
             + result['stage0'].get('fix_tests', 'not recorded'), '',
             '## Arms: image-side AMI (k=20)', '',
             '| Arm | Emotion | Style | Genre | Established |', '|---|---:|---:|---:|---|']
    for arm, r in result['arms'].items():
        values = [fmt(r['img'][a]) if r['img'] else '—' for a in LABELS]
        status = 'yes' if r['established'] else 'not established: ' + '; '.join(r['reasons'])
        lines.append('| ' + arm + ' | ' + ' | '.join(values) + f' | {status} |')
    lines += ['', '## Per-seed image-side AMI (k=20)', '',
              '| Arm | Seed | Emotion | Style | Genre |', '|---|---:|---:|---:|---:|']
    for arm, r in result['arms'].items():
        for seed, v in r['seeds'].items():
            if 'img' in v:
                lines.append(f'| {arm} | {seed} | ' + ' | '.join(fmt(v['img'][a]) for a in LABELS) + ' |')
    lines += ['', '## K-native AMI (argmax)', '',
              '| Arm | Side | Emotion | Style | Genre |', '|---|---|---:|---:|---:|']
    for arm, r in result['arms'].items():
        for key, side in (('native_img', 'Stage 2 image'), ('native_topic', 'Stage 1 painting topic')):
            values = [fmt(r[key][a]) if r[key] else '—' for a in LABELS]
            lines.append(f'| {arm} | {side} | ' + ' | '.join(values) + ' |')
    lines += ['', '## Kept topics', '',
              'A3: K = 40, the most-used 40 of 100 centers by fit-pair argmax count. '
              'Coverage uses final fit-caption topics; share before the cut uses all 100 centers. '
              'Over every available Stage 1 seed.', '',
              '| Arm | K | With >= 1 fit pair | With >= 1% | Share before cut |',
              '|---|---:|---:|---:|---:|']
    for arm, r in result['arms'].items():
        values = [str(int(r['K']['mean'])) if r['K'] else '—'] + [
            fmt(r['kept_topics'][key]) if r['kept_topics'] else '—'
            for key in ('with_fit_pair', 'with_1pct', 'share_before_cut')]
        lines.append(f'| {arm} | ' + ' | '.join(values) + ' |')
    lines += ['', '## Per-seed kept topics', '',
              '| Arm | Seed | K | With >= 1 fit pair | With >= 1% | Share before cut |',
              '|---|---:|---:|---:|---:|---:|']
    for arm, r in result['arms'].items():
        for seed, v in r['stage1_seeds'].items():
            coverage = v['kept_topics']
            lines.append(f"| {arm} | {seed} | {v['K']} | {coverage['with_fit_pair']} | "
                         f"{coverage['with_1pct']} | {coverage['share_before_cut']:.5f} |")
    lines += ['', '## K per gamma', '',
              'Algorithm 1, reported only (A3). Over every available Stage 1 seed.', '',
              '| Arm | 1.5 | 2 | 3 | 5 |', '|---|---:|---:|---:|---:|']
    for arm, r in result['arms'].items():
        values = [fmt(r['K_gamma'][g]) if r['K_gamma'] else '—' for g in ('1.5', '2', '3', '5')]
        lines.append(f'| {arm} | ' + ' | '.join(values) + ' |')
    lines += ['', '## Dev AUC', '', '| Arm | Macro AUC |', '|---|---:|']
    for arm, r in result['arms'].items():
        lines.append(f"| {arm} | {fmt(r['dev_auc'])} |")
    lines += ['', 'Algorithm 1 K(γ) columns below are reported only (A3); K is the trained count.', '',
              '| Arm | Seed | K | Algorithm 1 K(1.5) | Algorithm 1 K(2) | Algorithm 1 K(3) '
              '| Algorithm 1 K(5) | Dev AUC | Valid topics |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, r in result['arms'].items():
        for seed, v in r['seeds'].items():
            counts = ' | '.join(str(v['K_gamma'][g]) for g in ('1.5', '2', '3', '5'))
            auc = 'undefined' if v['dev_auc'] is None else f"{v['dev_auc']:.5f}"
            lines.append(f"| {arm} | {seed} | {v['K']} | {counts} | {auc} | {v['dev_valid_topics']} |")
    lines += ['', '## Row 1', '',
              'Captions: normalize each caption, painting mean, normalize. Image: normalized CLIP image.', '',
              '| Caption encoder | Aspect | CLIP image | Caption | Both-side mean |',
              '|---|---|---:|---:|---:|']
    for kind, r in result['row1'].items():
        for a in LABELS:
            lines.append(f"| {kind} | {a} | {fmt(r['img'][a])} | {fmt(r['cap'][a])} | {r['both_mean'][a]:.5f} |")
    lines += ['', '## Comparator', '',
              'D15 per aspect: among established arms, within 0.003 of the highest mean, '
              'choose the lowest arm number. Best of all arms uses the same rule. '
              'Incomplete three-seed arms have no aggregate and cannot be selected.', '',
              'Highest columns name the arm with the highest mean before the tie rule.', '',
              '| Aspect | Established: highest | Established comparator | Best of all: highest '
              '| Best of all arms |', '|---|---|---|---|---|']
    for a, picks in result['comparator'].items():
        cells = []
        for key in ('established', 'best_all'):
            pick = picks[key]
            top = '—' if pick is None else f"{pick['highest_arm']} ({pick['highest_mean']:.5f})"
            cell = '—' if pick is None else f"{pick['arm']} ({pick['mean']:.5f})"
            if pick is not None and not pick['established']:
                cell += ' (not established)'
            cells += [top, cell]
        lines.append(f'| {a} | ' + ' | '.join(cells) + ' |')
    if result['missing_outputs']:
        lines += ['', '## Missing outputs', '']
        lines += [f"- {name}: {', '.join(paths)}" for name, paths in result['missing_outputs'].items()]
    return '\n'.join(lines) + '\n'


def write_results(result, outdir=OUTPUTS):
    outdir = Path(outdir)
    text = markdown(result)
    atomic_json(outdir / 'pc_results.json', result)
    pending = outdir / 'pc_tables.pending.md'
    pending.write_text(text)
    pending.replace(outdir / 'pc_tables.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-row2', action='store_true')
    parser.add_argument('--outdir', type=Path, default=OUTPUTS)
    args = parser.parse_args()
    if args.check_row2:
        result = check_row2(outdir=args.outdir)
        print(result['status'], flush=True)
        return 0 if result['status'] == 'PASS' else 1
    s = load_frozen()
    labels = label_vectors(s, s['subset_pid'])
    result = score_outputs(s, labels, args.outdir)
    result['row1'] = row1(s, labels)
    write_results(result, args.outdir)
    print(args.outdir / 'pc_results.json', flush=True)
    return 0


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        sys.exit(main())
