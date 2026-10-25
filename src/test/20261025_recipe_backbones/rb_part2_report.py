"""Part 2 reported margins and diagnostics (rule §3.2 "Reported, deciding nothing"; spec §3.6). CPU.

Runs only after the stage's verdict file exists and writes `part2/reported_stage{3,4}.json` and `.md`. Nothing here
enters a verdict maximum or states an outcome. Per candidate Kk with an outcome in the verdict file:
- emotion: the point estimate is recomputed and must equal the verdict block; Fk (its frozen encoder image) and T
  (the instructed Qwen image) as extra seedless arms;
- style and genre: margins against (a) B1 to B6, (b) the matched arm (K4's is B1), (c) Fk, and the maximum of (a) and
  (b), plus T; every margin runs its own `default_rng(0)` bootstrap over exactly the arms it compares, in rule order,
  through `rb_score` (draws, indexed AMI scores, or its paired bootstrap);
- Leiden: the CT's 20-NN union graph and Leiden partitions (seeds 0 to 2) of every output set, as one r20-style
  reading per arm, margin against all comparators;
- each arm's seed spread, best epochs per student, and K1 recomputed with K4's entry set beside K4.
Also the CLIP B/32 store's cosine to the legacy CLIP store per painting (joined by painting name).
"""
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import rb_paths
import rb_encoders
import rb_score

BACKBONE = {'K2': 'siglip2_so400m', 'K3': 'clip_b32', 'K4': 'qwen3vl_emb_2b', 'K5': 'dinov2_large'}
FROZEN = {'K2': 'F2', 'K3': 'F3', 'K4': 'F4', 'K5': 'F5'}   # F4 = the default Qwen image (rule: F1 is A28 only)
ASPECTS = rb_paths.RB_ASPECTS
STAGES = rb_score.STAGES
COMPARATORS = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6')
LEIDEN_K = 20


def read_verdict(outputs, stage):
    """The first file boundary: refuse to run without the stage's verdict file."""
    path = Path(outputs) / f'part2/verdict_stage{stage}.json'
    assert path.is_file(), f'{path}: the reported margins run only after the verdict file exists'
    record = json.loads(path.read_text())
    assert record['stage'] == stage and record['scale'] == STAGES[stage], path
    return record, path


def held_labels():
    """(names, sha256, {aspect: int labels}) of the held set: the CT's list and annotation-order labels."""
    import ct_held
    names, sha, emotion = rb_score.held_setup()
    labels = ct_held.painting_labels(names)
    assert set(labels) == set(ASPECTS) and np.array_equal(labels['emotion'], emotion)
    return names, sha, {a: labels[a] for a in ASPECTS}


def frozen_image(candidate, names):
    """Fk: the candidate's encoder default image of the held paintings (float32 row / norm), by painting name."""
    return rb_encoders.full_images(BACKBONE[candidate], names)


def target_image(aspect, names, sha):
    """T: the CT's saved held instructed Qwen image of the aspect (ids checked against the held list)."""
    import ct_diag
    return ct_diag.target_images(aspect, names, sha)


def margin_arms(candidate):
    """Arm lists of the margins, each in rule order (Kk, B1 to B6, matched arm); K4's matched arm is B1."""
    arms = rb_score.arms_for(candidate)
    matched = arms[1 + len(COMPARATORS):] or ('B1',)
    return dict(a=(candidate,) + COMPARATORS, b=(candidate,) + matched, ab=arms)


def _bootstrap(fixed, labels, eligible, arms, n_resamples, workers):
    """`rb_score.paired_bootstrap` without its reading-shape asserts (Leiden gives every arm one r20 reading).
    Same draws (`draw_bootstrap`), same indexed score (`score_indices`), same worker functions."""
    draws = rb_score.draw_bootstrap(len(labels), eligible, arms, n_resamples)
    point = rb_score.score_indices(fixed, labels, np.arange(len(labels)), eligible, arms)
    if workers == 1:
        differences = [rb_score.score_indices(fixed, labels, rows, seeds, arms)['d'] for rows, seeds in draws]
    else:
        with multiprocessing.Pool(workers, initializer=rb_score._init_worker,
                                  initargs=(fixed, labels, arms)) as pool:
            differences = pool.map(rb_score._draw_difference, draws)
    point.update(interval=np.percentile(differences, [2.5, 97.5]).tolist(), bootstrap_d=differences)
    return point


def margin(fixed, labels, eligible, arms, n_resamples, workers, seedless=(), lenient=False):
    """One reported margin: its own default_rng(0) bootstrap over `arms`; d = A_cand - max entry; no outcome."""
    eligible = {arm: list(eligible[arm]) for arm in arms if arm not in seedless}
    if lenient:
        assert not seedless
        result = _bootstrap(fixed, labels, eligible, arms, n_resamples, workers)
    else:
        result = rb_score.paired_bootstrap(fixed, labels, eligible, arms, n_resamples, workers, seedless)
    entries = result['A_e']
    best = max(entries, key=entries.get)
    return dict(arms=list(arms), seedless=list(seedless), d=result['d'], interval=result['interval'],
                A_cand=result['A_cand'], A_e=entries, max_entry=best, n_resamples=n_resamples)


def seed_spread(point):
    """Per arm and reading: mean over the per-seed means of the partitions, and their min, max and sd (ddof 1)."""
    spread = {}
    for arm, seeds in point['per_seed'].items():
        spread[arm] = {}
        for reading in next(iter(seeds.values())):
            means = {int(s): float(np.mean(v[reading])) for s, v in sorted(seeds.items())}
            values = np.array(list(means.values()))
            spread[arm][reading] = dict(per_seed=means, mean=float(values.mean()), min=float(values.min()),
                                        max=float(values.max()),
                                        sd=float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    return spread


def best_epochs(scale, student, outputs):
    """Best epoch and best dev loss per aspect and seed, from the student's saved weight sidecars (K1: the CT's)."""
    root = (rb_score.CT_ROOT[scale] / 'd2/weights' if student == 'K1'
            else Path(outputs) / f'part2/{scale}/weights/{student}')
    result = {}
    for aspect in ASPECTS:
        result[aspect] = {}
        for seed in rb_score.SEEDS:
            meta = json.loads((root / f'{aspect}__s{seed}.json').read_text())
            result[aspect][str(seed)] = dict(best_epoch=int(meta['best_epoch']),
                                             best_dev_loss=float(meta['best_dev_loss']))
    return result


def frozen_partitions(x, candidate, folder, aspect, sha):
    """r20 partitions (three) of a seedless vector set, fixed once like an output; `candidate` only picks the K kind."""
    out = rb_score.fixed_partitions(x, candidate, folder, aspect, 0, sha, rb_score.array_sha256(x))
    return {0: out}


def leiden_reading(x):
    """The CT's 20-NN union graph (cosine, simplified) and three Leiden runs, int64 [3, n]."""
    import ct_diag
    return ct_diag.leiden_partitions(ct_diag.union_knn(x, LEIDEN_K))


def leiden_fixed(scale, arms, eligible, sources, aspect, outputs, sha, n_rows):
    """{arm: {seed: {'r20': Leiden partitions}}}, cached under part2/leiden/<scale>/<arm>/ with the source hash."""
    import ct_diag
    code = rb_score.sha_file(ct_diag.__file__)
    fixed = {}
    for arm in arms:
        fixed[arm] = {}
        key = aspect if arm[0] == 'K' else 'percept'
        for seed in eligible[arm]:
            src = sources[arm][str(seed)]['output_sha256']
            path = Path(outputs) / f'part2/leiden/{scale}/{arm}/{key}__s{seed}.npz'
            identity = dict(held_sha256=sha, source_sha256=src, k=LEIDEN_K, code_sha256=code)
            meta_path = path.with_suffix('.json')
            if path.exists():
                meta = json.loads(meta_path.read_text())
                assert meta['identity'] == identity and meta['partitions_sha256'] == rb_score.sha_file(path), path
                with np.load(path, allow_pickle=False) as z:
                    labels = z['labels']
            else:
                x = rb_score.load_output(Path(sources[arm][str(seed)]['path']), arm, n_rows, sha, scale, seed,
                                         aspect if arm[0] == 'K' else 'percept')
                labels = leiden_reading(x)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.pending.npz')
                np.savez_compressed(tmp, labels=labels)
                tmp.replace(path)
                meta_path.write_text(json.dumps(dict(identity=identity, partitions_sha256=rb_score.sha_file(path)),
                                                indent=2) + '\n')
            assert labels.shape == (3, n_rows) and np.issubdtype(labels.dtype, np.integer)
            fixed[arm][seed] = dict(r20=labels)
    return fixed


def report_candidate(candidate, block, ctx):
    """The reported block of one candidate (see the module docstring)."""
    scale, outputs, sha, n_rows = ctx['scale'], ctx['outputs'], ctx['sha'], ctx['n_rows']
    labels, nr, workers = ctx['labels'], ctx['n_resamples'], ctx['workers']
    arms = rb_score.arms_for(candidate)
    fixed, eligible, sources = rb_score.prepare_arms(scale, arms, ctx['rcol'], sha, outputs, n_rows=n_rows)
    assert eligible == block['eligibility']['eligible_seeds'], f'{candidate}: eligibility differs from the verdict'
    assert sources['outputs'] == block['sources']['held_outputs'], f'{candidate}: outputs changed since the verdict'
    everything = np.arange(n_rows)
    point = rb_score.score_indices(fixed, labels['emotion'], everything, eligible, arms)
    assert abs(point['d'] - block['d']) < 1e-9 and abs(point['A_cand'] - block['A_cand']) < 1e-9, \
        f'{candidate}: the recomputed emotion statistic differs from the verdict'
    groups = margin_arms(candidate)
    frozen = FROZEN[candidate]
    x = frozen_image(candidate, ctx['names'])
    assert x.shape[0] == n_rows
    folder = Path(outputs) / f'part2/partitions/{scale}'
    fixed[frozen] = frozen_partitions(x, candidate, folder / frozen, 'frozen', sha)
    report = dict(candidate=candidate, arms=list(arms), eligible_seeds=eligible, frozen_arm=frozen,
                  verdict=dict(d=block['d'], interval=block['interval'], outcome_id=block['outcome_id']),
                  aspects={}, best_epochs=best_epochs(scale, candidate, outputs))
    for aspect in ASPECTS:
        y = labels[aspect]
        if aspect == 'emotion':
            fx = fixed
        else:
            fx = dict(fixed)
            one, _, src = rb_score.prepare_arms(scale, (candidate,), ctx['rcol'], sha, outputs, aspect, n_rows)
            fx[candidate] = one[candidate]
            sources['outputs'][candidate] = src['outputs'][candidate]
        t = target_image(aspect, ctx['names'], sha)
        fx['T'] = frozen_partitions(t, candidate, folder / 'T', aspect, sha)
        res = dict(labels_masked=int((y < 0).sum()), margins={}, point_spread=seed_spread(
            rb_score.score_indices(fx, y, everything, eligible, arms)))
        m = res['margins']
        if aspect != 'emotion':
            m['a_B1_to_B6'] = margin(fx, y, eligible, groups['a'], nr, workers)
            m['b_matched'] = margin(fx, y, eligible, groups['b'], nr, workers)
            m['max_a_b'] = margin(fx, y, eligible, groups['ab'], nr, workers)
        m['c_frozen'] = margin(fx, y, eligible, (candidate, frozen), nr, workers, seedless=(frozen,))
        m['T'] = margin(fx, y, eligible, (candidate, 'T'), nr, workers, seedless=('T',))
        lf = leiden_fixed(scale, arms, eligible, sources['outputs'], aspect, outputs, sha, n_rows)
        m['leiden_vs_all'] = margin(lf, y, eligible, arms, nr, workers, lenient=True)
        report['aspects'][aspect] = res
        print(f'{candidate} {aspect}: ' + ', '.join(f'{k} d={v["d"]:.4f}' for k, v in m.items()), flush=True)
    return report, fixed, eligible


def report_k1(k4_fixed, k4_eligible, ctx):
    """K1's statistic with K4's entry set (B1 to B6), from the CT's saved K1 outputs, with its own bootstrap."""
    arms = rb_score.arms_for('K1')
    one, eligible_k1, sources = rb_score.prepare_arms(ctx['scale'], ('K1',), ctx['rcol'], ctx['sha'],
                                                      ctx['outputs'], n_rows=ctx['n_rows'])
    fixed = {'K1': one['K1'], **{arm: k4_fixed[arm] for arm in COMPARATORS}}
    eligible = {'K1': eligible_k1['K1'], **{arm: k4_eligible[arm] for arm in COMPARATORS}}
    result = rb_score.paired_bootstrap(fixed, ctx['labels']['emotion'], eligible, arms, ctx['n_resamples'],
                                       ctx['workers'])
    entries = result['A_e']
    return dict(arms=list(arms), d=result['d'], interval=result['interval'], A_cand=result['A_cand'], A_e=entries,
                max_entry=max(entries, key=entries.get), n_resamples=ctx['n_resamples'],
                eligible_seeds=eligible, best_epochs=best_epochs(ctx['scale'], 'K1', ctx['outputs']))


def clip_pair(names):
    """(new store rows, legacy rows) of the CLIP B/32 image of `names`, both float32 unit norm, joined by name."""
    import ct_held
    return rb_encoders.full_images('clip_b32', names), ct_held.clip_global_images(names)


def clip_names():
    """The painting names of the new CLIP B/32 store, sorted."""
    index = json.loads((rb_encoders.FULL_ROOT / 'clip_b32' / 'index.json').read_text())
    names = sorted(row['painting'] for row in index['images'])
    assert len(names) == len(set(names)) == rb_encoders.FULL_COUNT
    return names


def clip_cosine(outputs):
    """Per-painting cosine of the new CLIP B/32 store to the legacy CLIP image (D26); the array is saved beside the
    reports (`part2/clip_b32_cosine.npz`: names, cosine) and summarized."""
    names = clip_names()
    new, old = clip_pair(names)
    assert new.shape == old.shape and new.shape[0] == len(names) and new.ndim == 2
    cos = np.sum(new.astype(np.float64) * old.astype(np.float64), axis=1)
    assert np.isfinite(cos).all()
    path = Path(outputs) / 'part2/clip_b32_cosine.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert z['names'].tolist() == names and np.array_equal(z['cosine'], cos), f'{path}: changed'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix('.pending.npz'), names=np.asarray(names), cosine=cos)
        path.with_suffix('.pending.npz').replace(path)
    q = np.percentile(cos, [0, 1, 5, 50, 95, 100])
    return dict(n=len(names), mean=float(cos.mean()), sd=float(cos.std()),
                quantiles=dict(zip(['min', 'p1', 'p5', 'median', 'p95', 'max'], map(float, q))),
                below_0_99=int((cos < .99).sum()), below_0_999=int((cos < .999).sum()),
                array=str(path), array_sha256=rb_score.sha_file(path),
                legacy='ct_held.clip_global_images (first extraction ID of the painting, L2)',
                new='PRE/artelingo_full/clip_b32, row / float32 norm')


def _fmt(m):
    lo, hi = m['interval']
    return f'{m["d"]:+.4f} [{lo:+.4f}, {hi:+.4f}]'


def markdown(stage, scale, reports, k1, cosine, skipped):
    L = [f'# Part 2 reported margins, stage {stage} ({scale} scale)', '',
         'Decides nothing: no margin here enters a verdict maximum, and no outcome is stated. d = score of the '
         'candidate minus the largest comparator entry, with the 95% percentile interval of its own '
         '`default_rng(0)` bootstrap over exactly the arms listed. Emotion margins (a) and (b) are the verdict.', '']
    if skipped:
        L += [f'Skipped (no outcome in the verdict file): {", ".join(sorted(skipped))}.', '']
    for c, r in reports.items():
        L += [f'## {c}', '', f'Verdict (emotion, all entries): d {r["verdict"]["d"]:+.4f}, '
              f'{r["verdict"]["outcome_id"]}. Frozen encoder image: {r["frozen_arm"]}.', '',
              '| aspect | margin | d [interval] | candidate score | largest entry |', '|---|---|---|---|---|']
        for aspect, res in r['aspects'].items():
            for name, m in res['margins'].items():
                L.append(f'| {aspect} | {name} | {_fmt(m)} | {m["A_cand"]:.4f} | '
                         f'{m["max_entry"]} {m["A_e"][m["max_entry"]]:.4f} |')
        L += ['', 'Seed spread of the emotion score (mean, min, max, sd over seeds):', '',
              '| arm | reading | mean | min | max | sd |', '|---|---|---|---|---|---|']
        for arm, readings in r['aspects']['emotion']['point_spread'].items():
            for reading, s in readings.items():
                L.append(f'| {arm} | {reading} | {s["mean"]:.4f} | {s["min"]:.4f} | {s["max"]:.4f} | {s["sd"]:.4f} |')
        L += ['', 'Best epoch per seed (emotion, style, genre):', '']
        for aspect, seeds in r['best_epochs'].items():
            L.append(f'- {aspect}: ' + ', '.join(f's{s} {v["best_epoch"]}' for s, v in seeds.items()))
        L.append('')
    if k1:
        L += ['## K1 with K4\'s entry set (B1 to B6)', '',
              f'K1: d {_fmt(k1)}, score {k1["A_cand"]:.4f}, largest entry {k1["max_entry"]} '
              f'{k1["A_e"][k1["max_entry"]]:.4f}.']
        if 'K4' in reports:
            L.append(f'K4 beside it (verdict): d {reports["K4"]["verdict"]["d"]:+.4f} '
                     f'[{reports["K4"]["verdict"]["interval"][0]:+.4f}, {reports["K4"]["verdict"]["interval"][1]:+.4f}].')
        L += ['', 'K1 best epochs: ' + '; '.join(
            f'{a} ' + ', '.join(str(v['best_epoch']) for v in seeds.values()) for a, seeds in k1['best_epochs'].items()),
              '']
    q = cosine['quantiles']
    L += ['## CLIP B/32 store against the legacy CLIP store', '',
          f'{cosine["n"]} paintings, joined by painting name. Cosine mean {cosine["mean"]:.6f}, min {q["min"]:.6f}, '
          f'1st percentile {q["p1"]:.6f}, median {q["median"]:.6f}; {cosine["below_0_99"]} below 0.99 and '
          f'{cosine["below_0_999"]} below 0.999. Per painting: `{cosine["array"]}`.', '']
    return '\n'.join(L)


OTHER_BLOCKS = ('K1_with_K4_entry_set', 'clip_b32_cosine')


def block_sha(block):
    """Hash of one candidate's block as read from the verdict file; the reported block carries it."""
    return hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()


def run(stage, outputs=None, workers=8, n_resamples=2000, candidates=None, replace=None):
    outputs = Path(rb_paths.OUTPUTS if outputs is None else outputs)
    scale = STAGES[stage]
    verdict, verdict_path = read_verdict(outputs, stage)
    names, sha, labels = held_labels()
    assert verdict['held_sha256'] == sha
    rcol, rcol_path = rb_score.read_rcol_eligible(scale)
    blocks = verdict['blocks']
    done = sorted(c for c, b in blocks.items() if b.get('outcome_id'))
    path = outputs / f'part2/reported_stage{stage}.json'
    saved = json.loads(path.read_text())['blocks'] if path.exists() else {}
    if candidates:
        todo = sorted(candidates)
    else:                                   # a rerun after a deferral reports only the candidates not yet reported
        todo = [c for c in done if c not in saved]
    assert set(todo) <= set(done), f'no outcome in the verdict file for {sorted(set(todo) - set(done))}'
    skipped = sorted(c for c in blocks if c not in done)
    header = dict(stage=stage, scale=scale, held_sha256=sha)
    if not todo and not saved:              # no candidate has an outcome (each failed V3 or V4): nothing to report
        print(f'stage {stage}: no candidate has an outcome in the verdict file; nothing to report')
        return None
    if not todo:
        print(f'stage {stage}: every candidate with an outcome is already reported; nothing recomputed')
        record = json.loads(path.read_text())
    else:
        ctx = dict(scale=scale, outputs=outputs, sha=sha, names=names, n_rows=len(names), labels=labels, rcol=rcol,
                   n_resamples=n_resamples, workers=workers)
        reports, k1, kept = {}, None, {}
        for c in todo:
            reports[c], fixed, eligible = report_candidate(c, blocks[c], ctx)
            kept[c] = (fixed, eligible)
        if 'K4' in reports:
            k1 = report_k1(*kept['K4'], ctx)
        code = dict(rb_part2_report_sha256=rb_score.sha_file(__file__), **rb_score._code_identity())
        out = {c: dict(r, code=code, verdict_block_sha256=block_sha(blocks[c])) for c, r in reports.items()}
        if k1:
            out['K1_with_K4_entry_set'] = k1
        if 'clip_b32_cosine' not in saved:
            out['clip_b32_cosine'] = clip_cosine(outputs)
        record = rb_score.write_verdict(path, header, out, replace)
    shown = {c: b for c, b in sorted(record['blocks'].items()) if c not in OTHER_BLOCKS}
    md = markdown(stage, scale, shown, record['blocks'].get('K1_with_K4_entry_set'),
                  record['blocks']['clip_b32_cosine'], skipped)
    (outputs / f'part2/reported_stage{stage}.md').write_text(md + '\n')
    return record


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    p.add_argument('--smoke', action='store_true', help='20 draws; outputs must be under /tmp/claude-0/')
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    p.add_argument('--stage', type=int, choices=sorted(STAGES), required=True)
    p.add_argument('--candidates', nargs='+', choices=('K2', 'K3', 'K4', 'K5'),
                   help='default: every candidate with an outcome in the verdict file and no reported block yet')
    p.add_argument('--replace', metavar='REASON', help='V-check reason for replacing a changed block')
    args = p.parse_args(argv)
    if args.smoke:
        assert str(args.outputs.resolve()).startswith('/tmp/claude-0/'), 'smoke needs --outputs under /tmp/claude-0/'
    run(args.stage, args.outputs, args.workers, 20 if args.smoke else 2000, args.candidates, args.replace)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
