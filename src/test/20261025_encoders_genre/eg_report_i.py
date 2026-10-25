"""Stage i reported readings (ticket 06; rule §3.1, RB rule §3.2 "Reported, deciding nothing"; spec §3.6): K5 to K8.

A copy of `rb_part2_report`'s driver for this round's candidates (arm lists, output paths and the roots are this
round's; no K1 recompute with K4's entry set, no CLIP-only cosine diagnostic), reusing its helpers for the margins,
the seed spreads, the best epochs, the frozen-image partitions and the Leiden reading. Stage 5 is a copy of
`rb_a28.run_reported` with the same changes. Nothing here enters a verdict maximum or states an outcome.

It refuses to run before the stage's verdict file holds the candidate's outcome and V2 (`eg_v2`) agrees with that
block (RB rule §4.2, last bullet). Writes `<outputs>/part2/reported_stage{3,4,5}.{json,md}`.

CLI (from the round folder):
  python eg_report_i.py --stage 3|4|5 [--candidates K6 K7 ...] [--outputs DIR] [--smoke] [--workers 8] [--replace REASON]
"""
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import eg_paths
import eg_encoders
import eg_score
import eg_v2

import rb_a28  # noqa: E402
import rb_part2_report as rpr  # noqa: E402
import rb_score  # noqa: E402
import rb_v2  # noqa: E402

ASPECTS = rpr.ASPECTS
COMPARATORS = rpr.COMPARATORS
OLD_CANDIDATES = eg_v2.OLD_CANDIDATES          # K2 to K4 only for the parity check (K1 has no frozen-image F1 here)
PARITY_CANDIDATES = ('K2', 'K3', 'K4')
KINDS = {3: 'sample', 4: 'full'}


# --- the V2 boundary -------------------------------------------------------------------------------------------------------

def require_v2(stage, outputs, candidates, verdict_path):
    """RB rule §4.2, last bullet: each candidate has a V2 entry that agrees and still equals the verdict block now in
    the file (a block replaced after V2 ran is refused). Assertion text names the candidate and the fault."""
    path = Path(outputs) / f'part2/rederive_v2_stage{stage}.json'
    assert path.is_file(), f'{path}: the reported readings run only after V2 (eg_v2.py --stage {stage})'
    doc = json.loads(path.read_text())
    assert doc['stage'] == stage and not doc['smoke'], f'{path}: not a V2 file of stage {stage} (or a smoke run)'
    for cand in candidates:
        entry = doc['candidates'].get(cand)
        assert entry is not None, f'{cand}: no V2 entry in {path.name}'
        assert entry['comparison']['agree'], f'{cand}: V2 does not agree: {entry["comparison"]["mismatches"][:2]}'
        now = rb_v2.compare(entry, rb_v2.read_block(verdict_path, cand), check_scale=stage == 5)
        assert now['agree'], f'{cand}: V2 was run on another block than the verdict file holds now: {now["mismatches"][:2]}'


# --- stages 3 and 4 ---------------------------------------------------------------------------------------------------------

def margin_arms(candidate):
    """Arm lists of the margins, each in rule order (Kk, B1 to B6, matched arm); K4's matched arm is B1."""
    arms = eg_score.arms_for(candidate)
    matched = arms[1 + len(COMPARATORS):] or ('B1',)
    return dict(a=(candidate,) + COMPARATORS, b=(candidate,) + matched, ab=arms)


def _hashes(sources):
    """{arm: {seed: output_sha256}} of a `sources` dict (path and status left out)."""
    return {arm: {seed: info['output_sha256'] for seed, info in per_seed.items()} for arm, per_seed in sources.items()}


def report_candidate(candidate, block, ctx):
    """The reported block of one candidate: `rpr.report_candidate` with this round's arm lists and roots."""
    scale, roots, sha, n_rows = ctx['scale'], ctx['roots'], ctx['sha'], ctx['n_rows']
    labels, nr, workers = ctx['labels'], ctx['n_resamples'], ctx['workers']
    arms = eg_score.arms_for(candidate)
    fixed, eligible, sources = eg_score.prepare_arms(scale, arms, ctx['rcol'], sha, roots, n_rows=n_rows)
    assert eligible == block['eligibility']['eligible_seeds'], f'{candidate}: eligibility differs from the verdict'
    assert _hashes(sources['outputs']) == _hashes(block['sources']['held_outputs']), \
        f'{candidate}: outputs changed since the verdict'      # by content hash: a copied or linked outputs folder passes
    everything = np.arange(n_rows)
    point = rb_score.score_indices(fixed, labels['emotion'], everything, eligible, arms)
    assert abs(point['d'] - block['d']) < 1e-9 and abs(point['A_cand'] - block['A_cand']) < 1e-9, \
        f'{candidate}: the recomputed emotion statistic differs from the verdict'
    groups = margin_arms(candidate)
    frozen = rpr.FROZEN[candidate]
    x = rpr.frozen_image(candidate, ctx['names'])
    assert x.shape[0] == n_rows
    folder = Path(roots.cache) / f'part2/partitions/{scale}'
    fixed[frozen] = rpr.frozen_partitions(x, candidate, folder / frozen, 'frozen', sha)
    report = dict(candidate=candidate, arms=list(arms), eligible_seeds=eligible, frozen_arm=frozen,
                  verdict=dict(d=block['d'], interval=block['interval'], outcome_id=block['outcome_id']),
                  aspects={}, best_epochs=rpr.best_epochs(scale, candidate, roots.read))
    for aspect in ASPECTS:
        y = labels[aspect]
        if aspect == 'emotion':
            fx = fixed
        else:
            fx = dict(fixed)
            one, _, src = eg_score.prepare_arms(scale, (candidate,), ctx['rcol'], sha, roots, aspect, n_rows)
            fx[candidate] = one[candidate]
            sources['outputs'][candidate] = src['outputs'][candidate]
        t = rpr.target_image(aspect, ctx['names'], sha)
        fx['T'] = rpr.frozen_partitions(t, candidate, folder / 'T', aspect, sha)
        res = dict(labels_masked=int((y < 0).sum()), margins={}, point_spread=rpr.seed_spread(
            rb_score.score_indices(fx, y, everything, eligible, arms)))
        m = res['margins']
        if aspect != 'emotion':
            m['a_B1_to_B6'] = rpr.margin(fx, y, eligible, groups['a'], nr, workers)
            m['b_matched'] = rpr.margin(fx, y, eligible, groups['b'], nr, workers)
            m['max_a_b'] = rpr.margin(fx, y, eligible, groups['ab'], nr, workers)
        m['c_frozen'] = rpr.margin(fx, y, eligible, (candidate, frozen), nr, workers, seedless=(frozen,))
        m['T'] = rpr.margin(fx, y, eligible, (candidate, 'T'), nr, workers, seedless=('T',))
        lf = rpr.leiden_fixed(scale, arms, eligible, sources['outputs'], aspect, roots.cache, sha, n_rows)
        m['leiden_vs_all'] = rpr.margin(lf, y, eligible, arms, nr, workers, lenient=True)
        report['aspects'][aspect] = res
        print(f'{candidate} {aspect}: ' + ', '.join(f'{k} d={v["d"]:.4f}' for k, v in m.items()), flush=True)
    return report


def markdown(stage, scale, reports, skipped):
    L = [f'# Stage i reported margins, stage {stage} ({scale} scale)', '',
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
                L.append(f'| {aspect} | {name} | {rpr._fmt(m)} | {m["A_cand"]:.4f} | '
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
    return '\n'.join(L)


def read_verdict(outputs, stage):
    """The first file boundary: the stage's verdict file (its header checked) and its path."""
    path = Path(outputs) / f'part2/verdict_stage{stage}.json'
    assert path.is_file(), f'{path}: the reported readings run only after the verdict file exists'
    record = json.loads(path.read_text())
    assert record['stage'] == stage, path
    return record, path


def run_held(stage, outputs, workers, n_resamples, candidates, replace, allow_old):
    scale = KINDS[stage]
    verdict, verdict_path = read_verdict(outputs, stage)
    assert verdict['scale'] == scale, verdict_path
    names, sha, labels = rpr.held_labels()
    assert verdict['held_sha256'] == sha
    rcol, _ = eg_score.read_rcol_eligible(scale)
    blocks = verdict['blocks']
    done = [c for c in eg_v2.candidates_with_outcome(verdict_path, None, allow_old)[0] if c != 'K1']
    path = outputs / f'part2/reported_stage{stage}.json'
    saved = json.loads(path.read_text())['blocks'] if path.exists() else {}
    todo = list(candidates) if candidates else [c for c in done if c not in saved]
    assert set(todo) <= set(done), f'no outcome in the verdict file for {sorted(set(todo) - set(done))}'
    skipped = sorted(c for c, b in blocks.items() if not b.get('outcome_id'))
    if todo:
        require_v2(stage, outputs, todo, verdict_path)
    header = dict(stage=stage, scale=scale, held_sha256=sha)
    if not todo and not saved:
        print(f'stage {stage}: no candidate has an outcome in the verdict file; nothing to report')
        return None
    if not todo:
        print(f'stage {stage}: every candidate with an outcome is already reported; nothing recomputed')
        record = json.loads(path.read_text())
    else:
        ctx = dict(scale=scale, roots=eg_score.Roots(outputs, outputs), sha=sha, names=names, n_rows=len(names),
                   labels=labels, rcol=rcol, n_resamples=n_resamples, workers=workers)
        code = dict(eg_report_i_sha256=eg_score.sha_file(__file__), **eg_score.code_identity())
        record = None
        for c in todo:
            report = report_candidate(c, blocks[c], ctx)
            block = dict(report, code=code, verdict_block_sha256=rpr.block_sha(blocks[c]))
            record = eg_score.write_block(path, header, c, block, replace)
    shown = {c: b for c, b in sorted(record['blocks'].items())}
    (outputs / f'part2/reported_stage{stage}.md').write_text(markdown(stage, scale, shown, skipped) + '\n')
    return record


# --- stage 5 (copy of rb_a28's reported items with this round's paths) ------------------------------------------------------

def aspect_report(model, aspect, scale, roots, setup, fixed_other, seeds, n_resamples, workers):
    """`rb_a28.aspect_report` with the partitions read through `eg_score.a28_partitions`."""
    labels = setup['labels'][aspect]
    matched = rb_a28.MATCHED[model]
    fixed = dict(fixed_other)
    fixed[model], _ = eg_score.a28_partitions(model, aspect, scale, roots, setup, seeds)
    out = dict(A=None)
    plan = {'vs_a': (model, 'B1', 'B6'), 'vs_b': (model, matched)}
    if matched != 'B1':
        plan['vs_max_ab'] = eg_score.arms_a28(model)
    for name, arms in plan.items():
        out[name] = rb_a28.margin(fixed, labels, arms, seeds, n_resamples, workers)
    fixed['F'] = rb_a28.frozen_fixed(setup['images'](rb_a28.ENCODER[model]))
    out['vs_Fk'] = rb_a28.margin(fixed, labels, (model, 'F'), seeds, n_resamples, workers, ('F',))
    if aspect == 'emotion':
        if rb_a28.ENCODER[model] != 'qwen3vl_emb_2b':
            fixed['F'] = rb_a28.frozen_fixed(setup['images']('qwen3vl_emb_2b'))
            out['vs_F1'] = rb_a28.margin(fixed, labels, (model, 'F'), seeds, n_resamples, workers, ('F',))
        fixed['F'] = rb_a28.frozen_fixed(setup['instructed'](rb_a28.INSTRUCTED[aspect]))
        out['vs_T'] = rb_a28.margin(fixed, labels, (model, 'F'), seeds, n_resamples, workers, ('F',))
    out['A'] = out['vs_a']['A']
    return out


def reported_block(model, verdict, roots, setup, seeds, n_resamples, workers):
    """`rb_a28.reported_block` with this round's paths."""
    scale = verdict['scale']
    arms = eg_score.arms_a28(model)
    others = {}
    for arm in arms[1:]:
        others[arm], _ = eg_score.a28_partitions(arm, 'emotion', scale, roots, setup, seeds)
    block = dict(model=model, scale=scale, verdict_outcome=verdict['outcome_id'], verdict_d=verdict['d'],
                 verdict_interval=verdict['interval'], arms=list(arms))
    with threadpool_limits(limits=8):
        block['aspects'] = {aspect: aspect_report(model, aspect, scale, roots, setup, others, seeds, n_resamples,
                                                  workers) for aspect in rb_a28.ASPECTS}
        fixed = dict(others)
        fixed[model], _ = eg_score.a28_partitions(model, 'emotion', scale, roots, setup, seeds)
        held = setup['labels']['held']
        block['subsets'] = {}
        for name, idx in (('held_248', np.flatnonzero(held)), ('outside_498', np.flatnonzero(~held))):
            block['subsets'][name] = dict(n=int(len(idx)), **rb_a28.margin(
                rb_a28.subset(fixed, idx), setup['labels']['emotion'][idx], arms, seeds, n_resamples, workers))
        block['culture'] = rb_a28.culture_report(setup, {s: fixed[model][s]['r20'] for s in seeds})
    return block


def markdown5(blocks, skipped):
    L = ['# Stage i reported margins, stage 5 (ArtELingo-28)', '',
         'Decides nothing. d = score of the model minus the largest comparator entry, with the 95% percentile interval '
         'of its own `default_rng(0)` bootstrap over exactly the arms listed. Emotion `vs_a`/`vs_b`/`vs_max_ab` are '
         'the verdict\'s arms.', '']
    if skipped:
        L += [f'Skipped (no outcome in the verdict file): {", ".join(sorted(skipped))}.', '']
    for model, b in sorted(blocks.items()):
        L += [f'## {model} ({b["scale"]} scale)', '', f'Verdict: d {b["verdict_d"]:+.4f}, {b["verdict_outcome"]}.', '',
              '| aspect | margin | d [interval] | model score | arms |', '|---|---|---|---|---|']
        for aspect, res in b['aspects'].items():
            for name, m in res.items():
                if isinstance(m, dict):
                    L.append(f'| {aspect} | {name} | {rpr._fmt(m)} | {m["A"]:.4f} | {" ".join(m["arms"])} |')
        L += ['', '| subset | n | d [interval] | model score |', '|---|---|---|---|']
        for name, m in b['subsets'].items():
            L.append(f'| {name} | {m["n"]} | {rpr._fmt(m)} | {m["A"]:.4f} |')
        L += ['', f'Culture split (descriptive): mean AMI over languages {b["culture"]["mean_ami"]:.4f}.', '']
    return '\n'.join(L)


def run_a28(outputs, workers, n_resamples, seeds, candidates, replace, allow_old):
    setup = rb_a28.real_setup()
    verdict, verdict_path = read_verdict(outputs, 5)
    assert verdict['list_sha256'] == setup['sha256'] and verdict['n_rows'] == len(setup['names']), verdict_path
    done = [c for c in eg_v2.candidates_with_outcome(verdict_path, None, allow_old)[0] if c != 'K1']
    path = outputs / rb_a28.REPORTED
    saved = json.loads(path.read_text())['blocks'] if path.exists() else {}
    todo = list(candidates) if candidates else [c for c in done if c not in saved]
    assert set(todo) <= set(done), f'no outcome in the verdict file for {sorted(set(todo) - set(done))}'
    skipped = sorted(c for c, b in verdict['blocks'].items() if not b.get('outcome_id'))
    if todo:
        require_v2(5, outputs, todo, verdict_path)
    record = json.loads(path.read_text()) if path.exists() else None
    roots = eg_score.Roots(outputs, outputs)
    for model in todo:
        block = reported_block(model, verdict['blocks'][model], roots, setup, list(seeds), n_resamples, workers)
        block['code'] = dict(eg_report_i_sha256=eg_score.sha_file(__file__), **eg_score.code_identity())
        record = eg_score.write_block(path, rb_a28.header_of(setup), model, block, replace)
        print(f'{model}: reported', flush=True)
    if record is None:
        print('stage 5: no model has an outcome in the verdict file; nothing to report')
        return None
    (outputs / 'part2/reported_stage5.md').write_text(markdown5(record['blocks'], skipped) + '\n')
    return record


def run(stage, outputs=None, workers=8, n_resamples=2000, candidates=None, replace=None, seeds=eg_score.SEEDS,
        allow_old=False):
    outputs = Path(eg_paths.OUTPUTS if outputs is None else outputs)
    if stage == 5:
        return run_a28(outputs, workers, n_resamples, seeds, candidates, replace, allow_old)
    return run_held(stage, outputs, workers, n_resamples, candidates, replace, allow_old)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', type=int, choices=(3, 4, 5), required=True)
    p.add_argument('--candidates', nargs='+', choices=eg_encoders.CANDIDATES + PARITY_CANDIDATES,
                   help='default: every candidate with an outcome and no reported block yet. K2 to K4 only with '
                        '--outputs under /tmp/claude-0/ (the parity check on 12-24\'s saved outputs)')
    p.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    p.add_argument('--smoke', action='store_true', help='20 draws (stage 5: seed 0); outputs under /tmp/claude-0/')
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    p.add_argument('--replace', metavar='REASON', help='V-check reason for replacing a changed block')
    args = p.parse_args(argv)
    old = bool(args.candidates) and any(c in PARITY_CANDIDATES for c in args.candidates)
    if args.smoke or old:
        assert eg_paths.smoke_ok(args.outputs), 'smoke (and K2 to K4) need --outputs under /tmp/claude-0/'
    with threadpool_limits(limits=8):
        run(args.stage, args.outputs, args.workers, 20 if args.smoke else 2000, args.candidates, args.replace,
            seeds=[0] if args.smoke else eg_score.SEEDS, allow_old=old)


if __name__ == '__main__':
    main()
