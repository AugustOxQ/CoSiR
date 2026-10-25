"""Stage i scorer (ticket 05): verdicts at stages 3 to 5 for K6, K7, K8, K5, and the scorer half of X1 and X2.

The statistic is the RB rule's (part 2 and stage 5, unchanged): `rb_score`'s kernels (`load_output`,
`fixed_partitions`, `paired_bootstrap`, `candidate_block`, `write_verdict`) and `rb_a28`'s reading are reused as
they are. What is copied here is what hard-codes 12-24's lists: the arm lists, the held and ArtELingo-28 output
paths, the prepare/driver bodies that call them (code map §2). Outputs are read from a read root (`--outputs`; for
the checks, 12-24's saved outputs) and partition caches go under the cache root (always `--outputs`).
"""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import eg_paths
import eg_encoders
import eg_run

eg_encoders.extend_rb_maps()
import rb_a28  # noqa: E402
import rb_score  # noqa: E402

SEEDS = rb_score.SEEDS
STAGES = rb_score.STAGES                      # {3: 'sample', 4: 'full'}
NEW_STUDENTS = tuple(e.student for e in eg_encoders.ENCODERS if e.new)
MATCHED = {e.student: e.own for e in eg_encoders.ENCODERS if e.new}
OLD_MATCHED = {'K2': 'M2', 'K3': 'M3'}
ALL_MATCHED = dict(OLD_MATCHED, **MATCHED)
HELD_STUDENTS = ('K2', 'K3', 'K4') + NEW_STUDENTS
HELD_MATCHED = tuple(ALL_MATCHED.values())


def arms_for(candidate):
    """Rule §2.1 arm order: the candidate, B1 to B6, then its matched arm (K1 and K4 have none beyond B1).
    K1 to K4 are 12-24's lists (the checks use them)."""
    if candidate in ('K1', 'K2', 'K3', 'K4'):
        return rb_score.arms_for(candidate)
    assert candidate in MATCHED, candidate
    return (candidate, 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', MATCHED[candidate])


def output_path(scale, arm, seed, aspect='emotion', outputs=None):
    """Saved held output of an arm: the CT's files for K1 and B1 to B6, `<outputs>/part2/<scale>/held/...` for the
    students K2 to K8 (outputs = the read root) and the matched arms M2, M3, M5 to M8."""
    assert scale in rb_score.CT_ROOT and seed in SEEDS
    if arm in rb_score.CT_STEMS:
        return rb_score.output_path(scale, arm, seed, aspect)
    outputs = Path(eg_paths.OUTPUTS if outputs is None else outputs)
    if arm in HELD_STUDENTS:
        return outputs / f'part2/{scale}/held/{arm}/{aspect}__s{seed}.npz'
    assert arm in HELD_MATCHED, arm
    return outputs / f'part2/{scale}/held/{arm}/P3__s{seed}.npz'


# --- roots, arms, stage 3 and 4 ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Roots:
    """Where outputs are read and where this scorer's caches go. `read`: the students' and matched arms' outputs
    (`<read>/part2/<scale>/...`); `cache`: partitions (`<cache>/part2/partitions/...`); `frozen`: 12-24's full-scale
    B1 and B6 outputs on ArtELingo-28 (stage 5 only). Production reads and caches under `--outputs`; the X1 and X2
    checks read 12-24's saved outputs and cache under `--outputs`."""
    read: Path
    cache: Path
    frozen: Path = eg_paths.RB_OUT

    def __post_init__(self):
        for name in ('read', 'cache', 'frozen'):
            object.__setattr__(self, name, Path(getattr(self, name)))


held_setup = rb_score.held_setup
read_rcol_eligible = rb_score.read_rcol_eligible
sha_file = rb_score.sha_file


def code_identity():
    here = Path(__file__).resolve().parent
    return dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=here, text=True).strip(),
                eg_score_sha256=sha_file(__file__), rb_score_sha256=sha_file(rb_score.__file__),
                rb_a28_sha256=sha_file(rb_a28.__file__))


def check_v3_aspects(scale, candidate, roots, held_sha256, n_rows=rb_score.HELD_ROWS):
    """V3 on all three aspects of every seed of a candidate's output set (style and genre are only reported)."""
    for aspect in ('emotion', 'style', 'genre'):
        for seed in SEEDS:
            rb_score.load_output(output_path(scale, candidate, seed, aspect, roots.read), candidate, n_rows,
                                 held_sha256, scale, seed, aspect)


def prepare_arms(scale, arms, rcol_eligible, held_sha256, roots, aspect='emotion', n_rows=rb_score.HELD_ROWS):
    """`rb_score.prepare_arms` with this round's output paths and the cache root of `roots`: load and validate
    every needed output (V3), fix its partitions once; return (fixed, eligible, sources)."""
    eligible = rb_score.eligible_seeds(arms, rcol_eligible)
    nonfinite = {arm: [] for arm in rb_score.PAPER_ARMS}
    fixed, sources = {}, {}
    for arm in arms:
        fixed[arm], sources[arm] = {}, {}
        folder = roots.cache / f'part2/partitions/{scale}/{arm}'
        for seed in eligible[arm]:
            path = output_path(scale, arm, seed, aspect if arm[0] == 'K' else 'emotion', roots.read)
            assert path.exists(), f'missing eligible output: {path}'
            x = rb_score.load_output(path, arm, n_rows, held_sha256, scale, seed, aspect)
            sources[arm][str(seed)] = dict(path=str(path), output_sha256=sha_file(path))
            if x is None:
                nonfinite[arm].append(seed)
                sources[arm][str(seed)]['status'] = 'nonfinite_held'
                continue
            fixed[arm][seed] = rb_score.fixed_partitions(x, arm, folder, aspect if arm[0] == 'K' else 'percept', seed,
                                                         held_sha256, sources[arm][str(seed)]['output_sha256'])
    nonfinite = {arm: ss for arm, ss in nonfinite.items() if ss}
    return fixed, rb_score.eligible_seeds(arms, rcol_eligible, nonfinite), dict(outputs=sources, nonfinite_held=nonfinite)


def v5_failure(outputs, candidate, dataset):
    """Preserve a failed store's recorded cause instead of reporting a downstream missing output."""
    enc = eg_encoders.BY_STUDENT.get(candidate)
    path = Path(outputs) / 'stores/v5.json'
    if not enc or not enc.new or not path.is_file():
        return None
    key = f'{dataset}/{enc.store}'
    entry = json.loads(path.read_text()).get('stores', {}).get(key, {})
    if entry.get('pass') is not False:
        return None
    cause = entry.get('cause') or ', '.join(k for k, ok in entry.get('checks', {}).items() if not ok) or 'check failed'
    return f'V5 {key}: {cause}'


def stage_block(candidate, scale, roots, labels, sha, rcol, rcol_path, n_resamples, workers, arms=None):
    """One candidate's block at stage 3 or 4; a missing or invalid file (V3, V4) gives a block without an outcome."""
    arms = tuple(arms or arms_for(candidate))
    failure = v5_failure(roots.read, candidate, 'artelingo_full')
    if failure:
        return dict(candidate=candidate, arms=list(arms), scale=scale, outcome_id=None, v_failure=failure)
    try:
        check_v3_aspects(scale, candidate, roots, sha, len(labels))
        fixed, eligible, sources = prepare_arms(scale, arms, rcol, sha, roots, n_rows=len(labels))
        eligibility = dict(eligible_seeds=eligible, rcol_eligible_seeds=rcol, nonfinite_held=sources['nonfinite_held'])
        block = rb_score.candidate_block(
            fixed, labels, eligible, arms, scale, n_resamples, workers, eligibility,
            dict(held_outputs=sources['outputs'], rcol=dict(path=str(rcol_path), sha256=sha_file(rcol_path)),
                 code=code_identity()))
    except (AssertionError, FileNotFoundError) as err:
        block = dict(candidate=candidate, arms=list(arms), scale=scale, outcome_id=None,
                     v_failure=f'{type(err).__name__}: {err}')
    return block


def check_stage_3_done(candidate, outputs):
    """Rule §2.1 order: a candidate's stage 4 starts only after its stage 3 gave an outcome other than S1-NO."""
    path = Path(outputs) / 'part2/verdict_stage3.json'
    assert path.exists(), f'{candidate}: no stage 3 verdict ({path}); stage 4 starts after stage 3'
    outcome = json.loads(path.read_text())['blocks'].get(candidate, {}).get('outcome_id')
    assert outcome and outcome != 'S1-NO', f'{candidate}: stage 3 outcome is {outcome}; stage 4 does not run'


def check_sources(name, block, outputs, rcol=None):
    """A kept block is checked against its recorded inputs: every recorded `output_sha256` (a dict arm -> seed ->
    {path, output_sha256}) and the rcol file's `sha256` must equal the hash of the file now on disk. A block without
    an outcome has no sources and is not checked."""
    if block.get('outcome_id') is None:
        return
    files = [(info['path'], info['output_sha256']) for per_seed in outputs.values() for info in per_seed.values()]
    if rcol:
        files.append((rcol['path'], rcol['sha256']))
    assert files, f'{name}: a block with an outcome records no sources'
    changed = [path for path, recorded in files if sha_file(path) != recorded]
    assert not changed, f'{name}: kept block is stale, {len(changed)} recorded input(s) changed, first {changed[0]}; ' \
                        'give --replace REASON'


def write_block(path, header, name, block, replace):
    """`rb_score.write_verdict` for one block; its single `replacement` entry (reason, blocks, previous_sha256) stays as
    the latest replacement and gains `history`: every replacement made so far, each with its candidates, reason and
    the previous file hash (the header check of `write_verdict` ignores `replacement`)."""
    path = Path(path)
    before = json.loads(path.read_text()).get('replacement') if path.exists() else None
    record = rb_score.write_verdict(path, header, {name: block}, replace)
    now = record.get('replacement')
    if now is None or now == before:
        return record
    history = list((before or {}).get('history') or ([dict(before)] if before else []))
    history.append({k: v for k, v in now.items() if k != 'history'})
    record['replacement'] = dict(now, history=history)
    rb_score._atomic_json(path, record)
    return record


def _line(block):
    if block['outcome_id'] is None:
        return f'NO OUTCOME ({block["v_failure"]})'
    return f'd={block["d"]:.4f} interval={block["interval"]} {block["outcome_id"]}'


def run_stage(stage, candidates, outputs=None, workers=8, n_resamples=2000, replace=None, roots=None):
    """Stage 3 (sample) or 4 (full): each candidate's statistic into `<outputs>/part2/verdict_stage{3,4}.json`.

    No gate here (the gates V1 are `eg_run`'s) except rule §2.1's order: a candidate's stage 4 needs its stage 3 block
    with an outcome other than S1-NO. A candidate whose block is already in the file is kept unless `replace` gives a
    V-check reason (then it is recomputed and `write_verdict` records the change); a kept block whose recorded inputs
    changed since it was scored is refused (give `replace`)."""
    scale = STAGES[stage]
    outputs = Path(eg_paths.OUTPUTS if outputs is None else outputs)
    roots = roots or Roots(outputs, outputs)
    _, sha, labels = held_setup()
    rcol, rcol_path = read_rcol_eligible(scale)
    header = dict(stage=stage, scale=scale, held_sha256=sha)
    path = outputs / f'part2/verdict_stage{stage}.json'
    record = None
    for candidate in candidates:
        saved = json.loads(path.read_text())['blocks'] if path.exists() else {}
        if candidate in saved and not replace:
            check_sources(candidate, saved[candidate], saved[candidate].get('sources', {}).get('held_outputs', {}),
                          saved[candidate].get('sources', {}).get('rcol'))
            print(f'{candidate}: kept ({_line(saved[candidate])})', flush=True)
            record = json.loads(path.read_text())
            continue
        if stage == 4:
            check_stage_3_done(candidate, outputs)
        block = stage_block(candidate, scale, roots, labels, sha, rcol, rcol_path, n_resamples, workers)
        record = write_block(path, header, candidate, block, replace)
        print(f'{candidate}: {_line(block)}', flush=True)
    return record


# --- stage 5 ----------------------------------------------------------------------------------------------------------

OutputFault = rb_a28.OutputFault
A28_ASPECTS = rb_a28.ASPECTS
A28_VERDICT = rb_a28.VERDICT
X2_PARTS = ('student', 'matched', 'scorer')


def arms_a28(model):
    """Stage 5 entries (rule §2.1): the model, B1, B6, then its matched arm (K1 and K4 have none beyond B1)."""
    return rb_a28.arms_for(model)


def x2_not_passed(outputs):
    """None when X2 passed: all three regression files exist and pass. Else the reason, naming each missing or failing
    file (rule §4.1: stage 5 then runs at the sample scale)."""
    faults = []
    for part in X2_PARTS:
        path = Path(outputs) / f'part2/regression/x2_{part}.json'
        if not path.exists():
            faults.append(f'{path.name} missing')
        elif json.loads(path.read_text()).get('pass') is not True:
            faults.append(f'{path.name} fails')
    return 'X2 not passed: ' + ', '.join(faults) if faults else None


def scale_and_reason(model, outputs, stage4=None, check_x2=True):
    """(scale, reason) of a model's stage 5 (rule §2.1): full when its stage 4 block has an outcome, else sample;
    sample whenever X2 has not passed (§4.1: a missing X2 file counts). K1's stage 4 is the CT's stage 2: always full. `stage4` overrides the verdict
    file and `check_x2=False` skips the X2 files: for the checks on 12-24's outputs."""
    if model == 'K1':
        return rb_a28.scale_and_reason(model, outputs)
    if check_x2 and x2_not_passed(outputs):
        return 'sample', x2_not_passed(outputs)
    path = Path(stage4) if stage4 else Path(outputs) / 'part2/verdict_stage4.json'
    if path.exists() and json.loads(path.read_text())['blocks'].get(model, {}).get('outcome_id'):
        return 'full', None
    return 'sample', 'no stage 4 outcome'


def a28_output_path(roots, scale, arm, seed, aspect='emotion'):
    """`rb_a28.output_path` under the read root; B1 and B6 at full scale come from 12-24's `roots.frozen`
    (this round computes them only at the sample scale, ticket 04)."""
    root = roots.frozen if arm in ('B1', 'B6') and scale == 'full' else roots.read
    return rb_a28.output_path(root, scale, arm, seed, aspect)


def a28_partitions(arm, aspect, scale, roots, setup, seeds):
    """(fixed partitions per seed, sources): read, check and cluster one arm's outputs once (cached under the cache
    root)."""
    n, sha = len(setup['names']), setup['sha256']
    part_aspect = aspect if arm[0] == 'K' else 'percept'
    folder = roots.cache / f'part2/partitions/a28_{scale}/{arm}'
    fixed, sources = {}, {}
    for seed in seeds:
        path = a28_output_path(roots, scale, arm, seed, aspect)
        try:
            assert path.exists(), f'missing output: {path}'
            x = rb_a28.read_output(path, arm, scale, seed, part_aspect, sha, n)
        except (AssertionError, FileNotFoundError) as err:
            raise OutputFault(f'{type(err).__name__}: {err}') from err
        digest = sha_file(path)
        fixed[seed] = rb_score.fixed_partitions(x, arm, folder, part_aspect, seed, sha, digest)
        sources[str(seed)] = dict(path=str(path), output_sha256=digest)
    return fixed, sources


def a28_block(model, scale, roots, setup, seeds, n_resamples, workers):
    """`rb_a28.verdict_block` with this round's paths: V3 on all three aspects of the model, then the statistic."""
    arms = arms_a28(model)
    for aspect in A28_ASPECTS:
        a28_partitions(model, aspect, scale, roots, setup, seeds)
    fixed, sources = {}, {}
    for arm in arms:
        fixed[arm], sources[arm] = a28_partitions(arm, 'emotion', scale, roots, setup, seeds)
    eligible = {arm: list(seeds) for arm in arms}
    labels = setup['labels']['emotion']
    result = rb_score.paired_bootstrap(fixed, labels, eligible, arms, n_resamples, workers)
    values, counts = np.unique(labels, return_counts=True)
    return dict(candidate=model, arms=list(arms), scale=scale, **result, eligibility=dict(eligible_seeds=eligible),
                sources=sources, code=code_identity(),
                label_counts=dict(classes={str(v): int(c) for v, c in zip(values, counts)}, missing=0,
                                  ties=setup['labels']['n_ties']),
                outcome_id=rb_a28.outcome(result['d'], result['interval'][0]),
                bootstrap=dict(resamples=n_resamples, rng='numpy.random.default_rng(0) per model',
                               arm_order=list(arms), n_rows=len(labels), percentiles=[2.5, 97.5], method='linear'))


def model_block(model, scale, reason, roots, setup, seeds, n_resamples, workers):
    """One model's stage 5 block; a missing or invalid output file (V3) gives a block without an outcome."""
    failure = eg_run.stage3_failure(roots.read, model) if model in NEW_STUDENTS else None
    failure = ('no stage 3 outcome: ' + failure) if failure else v5_failure(roots.read, model, 'artelingo28')
    if failure:
        return dict(candidate=model, arms=list(arms_a28(model)), scale=scale, outcome_id=None,
                    v_failure=failure, **({'scale_reason': reason} if reason else {}))
    try:
        block = a28_block(model, scale, roots, setup, list(seeds), n_resamples, workers)
    except OutputFault as err:
        block = dict(candidate=model, arms=list(arms_a28(model)), scale=scale, outcome_id=None, v_failure=str(err))
    if reason:
        block['scale_reason'] = reason
    return block


def run_verdict(models, outputs=None, setup=None, workers=8, n_resamples=2000, seeds=SEEDS, replace=None, roots=None):
    """d_A28 and the reading of each model into `<outputs>/part2/verdict_stage5.json` (one block per model, at the
    scale `scale_and_reason` gives). A model whose block is already in the file is kept unless `replace` is given; a
    kept block is refused (give `replace`) when its scale is no longer the one due or a recorded input changed."""
    outputs = Path(eg_paths.OUTPUTS if outputs is None else outputs)
    roots = roots or Roots(outputs, outputs)
    setup = setup or rb_a28.real_setup()
    path = outputs / A28_VERDICT
    record = None
    for model in models:
        saved = json.loads(path.read_text())['blocks'] if path.exists() else {}
        scale, reason = scale_and_reason(model, outputs)
        if model in saved and not replace:
            assert saved[model]['scale'] == scale, \
                f'{model}: kept block is at the {saved[model]["scale"]} scale but {scale} is due; give --replace REASON'
            check_sources(model, saved[model], saved[model].get('sources', {}))
            print(f'{model}: kept ({_line(saved[model])})', flush=True)
            record = json.loads(path.read_text())
            continue
        block = model_block(model, scale, reason, roots, setup, seeds, n_resamples, workers)
        record = write_block(path, rb_a28.header_of(setup), model, block, replace)
        print(f'{model}: {_line(block)} ({scale})', flush=True)
    return record


# --- X1 and X2, scorer half --------------------------------------------------------------------------------------------

# (stage, models) with 12-24's arm lists: stage 3 and 4 held K2, K3, K4 (K4's matched arm is B1); stage 5 K1 to K4.
CHECKS = {'x1': ((3, ('K2', 'K3', 'K4')), (5, ('K1', 'K2', 'K3', 'K4'))), 'x2': ((4, ('K2', 'K3', 'K4')),)}


def _check_items(stage, model, block, saved):
    """Regression items of one block against 12-24's saved block: each A_e, A of the candidate, d, both bounds within
    1e-6, entries and arm list equal, outcome equal, and every bootstrap draw (a draw-order check)."""
    prefix = f'scorer/stage{stage}/{model}/'
    if block['outcome_id'] is None:
        return {prefix + 'outcome': {'pass': False, 'ours': None, 'ct': saved['outcome_id'],
                                     'cause': block['v_failure']}}
    items = rb_score.compare_with_ct(block, dict(saved, A_K1=saved['A_cand']))
    items = {prefix + name[len('scorer/'):]: item for name, item in items.items()}
    items[prefix + 'outcome'] = {'pass': block['outcome_id'] == saved['outcome_id'], 'ours': block['outcome_id'],
                                 'ct': saved['outcome_id']}
    items[prefix + 'arms'] = {'pass': block['arms'] == saved['arms'], 'ours': block['arms'], 'ct': saved['arms']}
    eligible = block['eligibility']['eligible_seeds']
    items[prefix + 'eligibility'] = {'pass': eligible == saved['eligibility']['eligible_seeds'], 'ours': eligible,
                                     'ct': saved['eligibility']['eligible_seeds']}
    return items


def run_check(which, outputs, workers=8, n_resamples=2000, frozen=None, setup=None):
    """Scorer half of X1 (`x1`: stages 3 and 5) or X2 (`x2`: stage 4): this scorer on 12-24's saved outputs
    (`frozen`/part2, read only) with 12-24's arm lists gives `frozen`/part2/verdict_stage{3,4,5}.json: each A_e, d,
    both bounds within 1e-6, outcomes equal. Caches go under `outputs`; items go into
    `<outputs>/part2/regression/<which>_scorer.json`."""
    frozen, outputs = Path(frozen or eg_paths.RB_OUT), Path(outputs)
    assert frozen.resolve() != outputs.resolve() and frozen.resolve() not in outputs.resolve().parents, \
        f'{outputs}: inside the frozen outputs {frozen}'
    roots = Roots(frozen, outputs, frozen)
    items = {}
    for stage, models in CHECKS[which]:
        verdict = json.loads((frozen / f'part2/verdict_stage{stage}.json').read_text())
        assert verdict['stage'] == stage
        if stage == 5:
            setup = setup or rb_a28.real_setup()
            assert verdict['list_sha256'] == setup['sha256'] and verdict['n_rows'] == len(setup['names'])
        else:
            _, sha, labels = held_setup()
            assert verdict['held_sha256'] == sha and verdict['scale'] == STAGES[stage]
            rcol, rcol_path = read_rcol_eligible(STAGES[stage])
        for model in models:
            saved = verdict['blocks'][model]
            assert saved['outcome_id'] and len(saved['bootstrap_d']) == n_resamples, \
                f'stage {stage} {model}: the saved block has {len(saved["bootstrap_d"])} draws, the check wants {n_resamples}'
            if stage == 5:
                scale, reason = scale_and_reason(model, frozen, check_x2=False)
                assert scale == saved['scale'], f'{model}: scale {scale} but 12-24 ran {saved["scale"]}'
                block = model_block(model, scale, reason, roots, setup, SEEDS, n_resamples, workers)
            else:
                block = stage_block(model, STAGES[stage], roots, labels, sha, rcol, rcol_path, n_resamples, workers,
                                    arms=saved['arms'])
                assert tuple(saved['arms']) == arms_for(model), (model, saved['arms'])
            items.update(_check_items(stage, model, block, saved))
            print(f'stage {stage} {model}: ours {block["outcome_id"]} 12-24 {saved["outcome_id"]}', flush=True)
    return rb_score.write_regression_items(outputs / f'part2/regression/{which}_scorer.json',
                                           dict(items, code={'pass': True, **code_identity()}))


# --- CLI ----------------------------------------------------------------------------------------------------------------

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    p.add_argument('--smoke', action='store_true', help='stages only: 20 draws (stage 5: seed 0 only); under /tmp/claude-0/')
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument('--check', choices=sorted(CHECKS), help='scorer half of X1 or X2 on 12-24\'s saved outputs')
    what.add_argument('--stage', type=int, choices=(3, 4, 5), help='verdict of stage 3 (sample), 4 (full) or 5 (A28)')
    p.add_argument('--candidates', nargs='+', choices=eg_encoders.CANDIDATES,
                   help='default: all four; required with --replace')
    p.add_argument('--replace', metavar='REASON', help='V-check reason for replacing a saved block')
    args = p.parse_args(argv)
    if args.replace and not args.candidates:
        p.error('--replace needs --candidates (one or more): a reason applies only to the candidates named')
    args.candidates = args.candidates or list(eg_encoders.CANDIDATES)
    if args.smoke:
        assert eg_paths.smoke_ok(args.outputs), 'smoke needs --outputs under /tmp/claude-0/'
        assert args.stage, 'a regression check is never shrunk: no --smoke with --check'
    if args.check:
        record = run_check(args.check, args.outputs, args.workers)
        bad = [k for k, v in record['items'].items() if not v['pass'] and k.startswith('scorer/')]
        print(f'{args.check}: pass={record["pass"]}; failing scorer items: {bad}', flush=True)
        raise SystemExit(0 if not bad else 1)
    kw = dict(workers=args.workers, n_resamples=20 if args.smoke else 2000, replace=args.replace)
    if args.stage == 5:
        record = run_verdict(args.candidates, args.outputs, seeds=[0] if args.smoke else SEEDS, **kw)
    else:
        record = run_stage(args.stage, args.candidates, args.outputs, **kw)
    failed = [c for c in args.candidates if record['blocks'][c].get('outcome_id') is None]
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
