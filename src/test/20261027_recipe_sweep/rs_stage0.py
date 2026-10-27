"""Stage 0 of the recipe sweep (rule §2.1, §3.1, §4.1): G's emotion claim at both scales, held inference of G's sample
seeds, and the gates V0a, V0c and V0d.

The statistic is the RB rule's part 2 statistic (12-24's `rb_score`, unchanged: `prepare_arms`'s comparator handling,
`eligible_seeds`, `paired_bootstrap`, `candidate_block` and the draw order) with a candidate whose held outputs come
from a folder given as a parameter (`Candidate`): G at both scales here, c* at full scale in ticket 09, 12-24's K4 in
the scorer check. The candidate is the first arm, then B1 to B6; its seeds are 0 to 4.

Files (all under --outputs, default `rs_paths.OUTPUTS`):
  stage0/sample/held/G/{aspect}__s{seed}.npz|json   held inference of G's sample heads on H (float16, 2048 wide, H's
                                                    order), as 12-24's `part2/sample/held/K4/`
  stage0/partitions/{scale}/G/                      the candidate's fixed r20 partitions (rb_score's format)
  stage0/regression/partitions/{scale}/K4/          the same for the scorer check on 12-24's K4 outputs
  gates/v0a.json (cpu part, gpu part), gates/v0c.json, gates/v0d.json
  stage0/verdict.json                               per scale d, A_G, each A_e, bounds, outcome, and E0

CLI (`python rs_stage0.py ...`):
  check v0a --part cpu    scorer check at both scales, G's full outputs against 12-27's stage C record, G's sample
                          weights recorded (CPU)
  check v0a --part gpu    held inference from G's full seed-0 weights and from R6-5's sample seed-1 weights, bit for bit
                          (GPU; run with CUBLAS_WORKSPACE_CONFIG=:4096:8)
  held --seeds 0 1 2 3 4  held inference of G's sample heads (GPU; needs gates/v0a.json, and for seed 4 the trained
                          run and gates/v1a_train.json)
  check v0c / check v0d   the gates over the written sets
  verdict                 stage0/verdict.json (needs v0a, v0c, v0d)
"""
import argparse
import hashlib
import importlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import rs_paths as paths
import rs_sheet as S
import rs_train
import rb_encoders
import rb_pick
import rb_score
import rb_student
import rg_train
import rg_variants as V

ASPECTS = tuple(paths.ASPECTS)
SCALES = ('sample', 'full')
SEEDS = (0, 1, 2, 3, 4)
SAMPLE_FROZEN_SEEDS = (0, 1, 2, 3)         # 12-27's GS-0.02 weights; seed 4 is this round's
BASELINES = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6')
G_LABEL = 'G'                              # G's arm label in the arms tuple and in the sample sidecars
ENCODER = 'qwen3vl_emb_2b'
RECIPE = V.BASE                            # 'R6-5'
KIND = {'sample': 'ES', 'full': 'EF'}      # G-ES-* and G-EF-* (rule §4.1)
TOL = 1e-6                                 # V0a
N_HELD, WIDTH, HELD_SHA256 = paths.N_HELD, paths.WIDTH, paths.HELD_SHA256
# What decides V0a to V0d: this module and the frozen modules that score, infer and load (gates carry their hashes).
CODE_MODULES = ('rs_stage0', 'rs_paths', 'rb_score', 'rb_student', 'rb_encoders', 'rb_pick', 'rb_ami', 'ct_score',
                'ct_held', 'pav_score', 'apc_data', 'ct2_data')
R6_5_S1 = paths.EG_OUT / 'stage_iii/train/R6-5/s1'                  # 12-25's R6-5 sample seed 1, equal to K4's
K4_SAMPLE = paths.RB_PART2 / 'sample'
G_FULL_HELD = paths.RG_FULL / 'held' / paths.G_ARM
G_FULL_WEIGHTS = paths.RG_FULL / 'weights' / paths.G_ARM
G_FULL_RECORD = paths.RG_STAGE_C / 'reported.json'                  # 12-27's stage C metadata (output SHA-256s)
RB_VERDICT = {'sample': paths.RB_PART2 / 'verdict_stage3.json', 'full': paths.RB_PART2 / 'verdict_stage4.json'}


class Refusal(RuntimeError):
    """A step that its gates or inputs do not allow."""


@dataclass(frozen=True)
class Candidate:
    """A candidate's held outputs: `label` is its arm in the arms tuple and the verdict block ('G'; 'K4' in the scorer
    check; c* in ticket 09); `held_dir` holds `{aspect}__s{seed}.npz|json`; `sidecar_arm` is the `arm` those sidecars
    record ('G' for this round's sample outputs, 'GS-0.02' for 12-27's full ones, 'K4' for 12-24's)."""
    label: str
    held_dir: Path
    sidecar_arm: str


def sample_g(outputs):
    return Candidate(G_LABEL, Path(outputs) / 'stage0/sample/held/G', G_LABEL)


FULL_G = Candidate(G_LABEL, G_FULL_HELD, paths.G_ARM)


def k4(scale):
    return Candidate('K4', paths.RB_PART2 / scale / 'held/K4', 'K4')


# ================================================================== the rule's outcome logic (pure)
def arms_for(label):
    """The candidate first, then B1 to B6 (rule §2.1; G reads the default Qwen image, so its matched arm is B1, which
    is already in the list: no other matched arm)."""
    return (label,) + BASELINES


def outcome_id(d, lower, scale):
    """G-ES-* (sample) or G-EF-* (full) from `rb_score.outcome`'s thresholds (rule §4.1, on unrounded values):
    GO d >= 0.01 and lower > 0; MM d >= 0.01 and lower <= 0; IN 0 < d < 0.01; NO d <= 0."""
    kind = rb_score.outcome(d, lower, scale).split('-')[1]
    assert kind in ('GO', 'MM', 'IN', 'NO')
    return f'G-{KIND[scale]}-{kind}'


def e0(outcomes):
    """E0, the licensed sentence (rule §4.1): G-EF-GO, and the sample scale has an outcome other than G-ES-NO."""
    assert set(outcomes) == set(SCALES)
    return outcomes['full'] == 'G-EF-GO' and outcomes['sample'] is not None and outcomes['sample'] != 'G-ES-NO'


# ================================================================== files and records
def sha_file(path):
    return rb_score.sha_file(path)


def _clean(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _atomic_json(path, value):
    rb_score._atomic_json(path, _clean(value))


def code_files():
    files = {m: Path(importlib.import_module(m).__file__) for m in CODE_MODULES}
    return {m: sha_file(path) for m, path in files.items()}


def code_sha256():
    return hashlib.sha256(''.join(code_files().values()).encode()).hexdigest()


def _git_state():
    def git(*args):
        return subprocess.run(['git', *args], cwd=paths.ROUND, capture_output=True, text=True, check=True).stdout.strip()
    # code_dirty: any tracked or untracked .py file of the repository differs from HEAD (as rs_graphs).
    return dict(code_commit=git('rev-parse', 'HEAD'), code_dirty=bool(git('status', '--porcelain', '--', ':/*.py')))


def gate_path(outputs, name):
    return Path(outputs) / 'gates' / f'{name}.json'


def read_gate(outputs, name):
    """A gate file that passes for the current code, or a Refusal with the reason."""
    path = gate_path(outputs, name)
    if not path.is_file():
        raise Refusal(f'{path} is missing: run its check first')
    gate = json.loads(path.read_text())
    if gate['code_sha256'] != code_sha256():
        raise Refusal(f'{path} was written by other stage 0 code: rerun the check')
    if not gate['pass']:
        raise Refusal(f'{path} failed: this step does not run')
    return gate


def read_v1a_train(outputs):
    """gates/v1a_train.json for seed 4: present, passing, and stamped with the trainer files that are on disk now."""
    path = gate_path(outputs, 'v1a_train')
    if not path.is_file():
        raise Refusal(f'{path} is missing: V1a (rs_train regress v1a) must pass before seed 4 trains')
    gate = json.loads(path.read_text())
    if gate['pass'] is not True:
        raise Refusal(f'{path} failed: seed 4 does not run')
    now = rs_train.code_sha256(training_only=True)
    stale = sorted(k for k, v in now.items() if gate['code_sha256'].get(k) != v)
    if stale:
        raise Refusal(f'{path} was written for other trainer code ({stale}): rerun V1a')
    return gate


def write_gate(outputs, name, passed, items, **extra):
    gate = dict(name=name, **{'pass': bool(passed)}, items=items, code_sha256=code_sha256(), code_files=code_files(),
                **_git_state(), **extra)
    _atomic_json(gate_path(outputs, name), gate)
    return gate


def _bits_equal(a, b):
    """Same dtype and shape and the same bits (float16 compared as uint16, so a NaN is a value)."""
    return (a.dtype == b.dtype and a.shape == b.shape
            and bool(np.array_equal(np.ascontiguousarray(a).view(np.uint16), np.ascontiguousarray(b).view(np.uint16))))


def _close(ours, theirs):
    diff = float(abs(ours - theirs))
    return {'pass': bool(diff < TOL), 'ours': float(ours), 'theirs': float(theirs), 'abs_diff': diff}


# ================================================================== a candidate's held outputs
def v0c_problem(x):
    """V0c on one output: None, or the first problem (12,281 finite unit-norm float16 rows, 2048 wide)."""
    return rb_pick.vp2_problem(x, N_HELD)


def failed_path(held_dir, aspect, seed):
    return Path(held_dir) / f'failed__{aspect}__s{seed}.json'


def load_candidate(cand, scale, aspect, seed):
    """One output of a candidate as float32 and the file's SHA-256: its sidecar binds the arm, scale, aspect, seed,
    row count, width, dtype and H's list hash, and V0c holds (rows, finite, unit norm)."""
    path = Path(cand.held_dir) / f'{aspect}__s{seed}.npz'
    assert path.is_file(), f'{path}: missing'
    assert not failed_path(cand.held_dir, aspect, seed).exists(), f'{path}: V0d, a failure record exists'
    arrays, info = rb_student.read_outputs(path)
    assert sorted(arrays) == ['img'], path
    expected = dict(kind='held', arm=cand.sidecar_arm, scale=scale, aspect=aspect, seed=seed, rows=N_HELD, width=WIDTH,
                    dtype='float16', list_sha256=HELD_SHA256)
    for key, value in expected.items():
        assert info.get(key) == value, f'{path}: sidecar {key} is {info.get(key)!r}, expected {value!r}'
    assert info['sha256'] == info['arrays']['img']['sha256'], path
    problem = v0c_problem(arrays['img'])
    assert problem is None, f'{path}: V0c: {problem}'
    return arrays['img'].astype(np.float32), sha_file(path)


def candidate_fixed(cand, scale, folder, aspect='emotion'):
    """Fixed r20 partitions of the candidate's five seeds and the sources record. `rb_score.fixed_partitions` picks the
    image-head reading (r20 only) from the first letter of the arm: 'K4' is that selector here, not the candidate."""
    fixed, sources = {}, {}
    for seed in SEEDS:
        x, sha = load_candidate(cand, scale, aspect, seed)
        sources[str(seed)] = dict(path=str(Path(cand.held_dir) / f'{aspect}__s{seed}.npz'), output_sha256=sha)
        fixed[seed] = rb_score.fixed_partitions(x, 'K4', folder, aspect, seed, HELD_SHA256, sha)
    return fixed, sources


def comparators(scale, rcol, rb_out=None):
    """B1 to B6's fixed partitions from 12-24's validated cache. A missing file would be written into the frozen
    folder by `prepare_arms`, so it is refused first."""
    rb_out = Path(paths.RB_OUT if rb_out is None else rb_out)
    eligible = rb_score.eligible_seeds(BASELINES, rcol)
    for arm in BASELINES:
        for seed in eligible[arm]:
            for reading in ('r20', 'own'):
                part = rb_out / f'part2/partitions/{scale}/{arm}/percept__s{seed}__{reading}.npz'
                if not part.is_file():
                    raise Refusal(f'{part} is missing: the frozen cache would be written to')
    return rb_score.prepare_arms(scale, BASELINES, rcol, HELD_SHA256, rb_out)


def assemble(label, cand_fixed, comp_fixed, rcol, nonfinite, labels, scale, n_resamples=2000, workers=8,
             sources=None):
    """The candidate's verdict block: arms in rule order, eligibility by `rb_score.eligible_seeds` (the candidate has
    seeds 0 to 4), then `rb_score.candidate_block` (its own default_rng(0): I_b, then S_b per arm with seeds)."""
    arms = arms_for(label)
    eligible = rb_score.eligible_seeds(arms, rcol, nonfinite)
    assert eligible[label] == list(SEEDS)
    assert set(cand_fixed) == set(SEEDS) and label not in comp_fixed
    fixed = dict(comp_fixed, **{label: cand_fixed})
    eligibility = dict(eligible_seeds=eligible, rcol_eligible_seeds=rcol, nonfinite_held=nonfinite)
    block = rb_score.candidate_block(fixed, labels, eligible, arms, scale, n_resamples, workers, eligibility,
                                     sources or {})
    block['outcome_id'] = outcome_id(block['d'], block['interval'][0], scale)
    return block


def score_scale(cand, scale, cache, n_resamples=2000, workers=8):
    """One scale's block for `cand`: its outputs and 12-24's B1 to B6 on the held emotion labels."""
    names, sha, labels = rb_score.held_setup()
    assert sha == HELD_SHA256
    rcol, rcol_path = rb_score.read_rcol_eligible(scale)
    comp_fixed, _, comp_sources = comparators(scale, rcol)
    cand_fixed, cand_sources = candidate_fixed(cand, scale, Path(cache) / scale / cand.label)
    sources = dict(held_outputs=dict(comp_sources['outputs'], **{cand.label: cand_sources}),
                   rcol=dict(path=str(rcol_path), sha256=sha_file(rcol_path)))
    return assemble(cand.label, cand_fixed, comp_fixed, rcol, comp_sources['nonfinite_held'], labels, scale,
                    n_resamples, workers, sources)


# ================================================================== held inference
def held_images():
    names, sha = rb_student.held_names()
    assert sha == HELD_SHA256
    x = rb_encoders.full_images(ENCODER, names)
    assert x.shape == (N_HELD, WIDTH) and x.dtype == np.float32
    return x


def infer(weights, aspect, x, device):
    """rb_student's inference of one saved best view (batches of 512, float16, checked)."""
    return rb_student.infer(weights, aspect, WIDTH, x, torch.device(device))


def weights_record(weights):
    """(device the head trained on, from its record) of a best state `weights` with a `.json` beside it."""
    return json.loads(Path(weights).with_suffix('.json').read_text())['config']['device']


def sample_weights(outputs, seed, aspect, recorded=None):
    """(path, SHA-256) of G's sample best state: seeds 0 to 3 are 12-27's files (their SHA-256 must equal the V0a
    record `recorded`, a dict `s{seed}/{aspect}` -> sha), seed 4 is this round's run `run_dir(outputs, 'G', 4)`,
    trained by the trainer V1a passed for."""
    assert seed in SEEDS and aspect in ASPECTS
    if seed in SAMPLE_FROZEN_SEEDS:
        folder = paths.RG_TRAIN / paths.G_ARM / f's{seed}'
        record = json.loads((folder / 'run.json').read_text())
        assert record['status'] == 'ok' and record['variant'] == paths.G_ARM and record['seed'] == seed, folder
    else:
        gate = read_v1a_train(outputs)
        folder = paths.run_dir(outputs, S.name(S.config()), seed)
        if not (folder / 'run.json').is_file():
            raise Refusal(f'{folder}: seed 4 is not trained (rs_train train --configs G --seeds 4)')
        record = json.loads((folder / 'run.json').read_text())
        if record['status'] != 'ok':
            raise Refusal(f'{folder}: the run is {record["status"]} ({record.get("cause")})')
        assert (record['name'], record['seed'], record['config'], record['smoke']) == (
            S.name(S.config()), seed, rs_train.jsonable(S.config()), False), f'{folder}: not G at seed 4'
        assert record['p_sha256'] == paths.P_SHA256, folder
        now = rs_train.code_sha256(training_only=True)
        assert all(record['code_sha256'].get(k) == v == gate['code_sha256'][k] for k, v in now.items()), \
            f'{folder}: trained by other code than V1a passed for'
    path = folder / f'{aspect}.pt'
    assert path.is_file(), path
    sha = sha_file(path)
    if recorded is not None and seed in SAMPLE_FROZEN_SEEDS:
        assert recorded[f's{seed}/{aspect}'] == sha, f'{path}: SHA-256 differs from the V0a record'
    return path, sha


def infer_set(held_dir, arm, scale, seeds, weights_of, x, device, *, recipe=RECIPE):
    """Held outputs of `arm`'s three best views for each seed into `held_dir` (rb_student's file format).

    `weights_of(seed, aspect)` -> (path, sha256). An existing file must match its meta, weights SHA-256 included (V0d: a
    retrained or replaced head is caught here), and pass V0c. V0c on a new output: a failure is rerun once from the same
    weights; a second failure is recorded in `failed__<aspect>__s<seed>.json` and stops (V0d). A seed with such a record
    is refused before anything is written."""
    held_dir = Path(held_dir)
    seeds = list(seeds)
    assert seeds and len(set(seeds)) == len(seeds) and set(seeds) <= set(SEEDS) and x.shape == (N_HELD, WIDTH)
    for seed in seeds:
        for aspect in ASPECTS:
            assert not failed_path(held_dir, aspect, seed).exists(), \
                f'V0d: {arm} {aspect} s{seed} failed V0c twice ({failed_path(held_dir, aspect, seed)}); no new attempt'
    weights = {(seed, aspect): weights_of(seed, aspect) for seed in seeds for aspect in ASPECTS}   # all before any write
    names, names_sha = rb_student.held_names()
    result = {}
    for seed in seeds:
        for aspect in ASPECTS:
            path_w, sha_w = weights[seed, aspect]
            recorded = torch.device(weights_record(path_w))
            assert torch.device(device) == recorded, 'inference device differs from training device'
            path = held_dir / f'{aspect}__s{seed}.npz'
            meta = dict(kind='held', arm=arm, scale=scale, aspect=aspect, seed=seed, encoder=ENCODER, rows=len(names),
                        width=WIDTH, dtype='float16', list_sha256=names_sha, weights_sha256=sha_w,
                        image_sha256=rb_student.sha_array(x), device=str(recorded), recipe=recipe,
                        switches=V.switches(paths.G_ARM))
            if path.exists():
                arrays, info = rb_student.read_outputs(path)
                assert all(info[k] == v for k, v in meta.items()), f'{path}: existing output differs from this run'
                out = arrays['img']
                problem = v0c_problem(out)
                assert problem is None, f'{path}: V0c: {problem}'
            else:
                problems = []
                for attempt in (1, 2):
                    try:
                        out = infer(path_w, aspect, x, recorded)
                        problem = v0c_problem(out)
                    except AssertionError as error:      # the inputs are checked above: this is the output's check
                        problem = f'infer: {error}'
                    if problem is None:
                        break
                    problems.append(problem)
                    print(f'{path}: V0c failure ({problem}); {"rerun once" if attempt == 1 else "stop"}', flush=True)
                else:
                    _atomic_json(failed_path(held_dir, aspect, seed),
                                 dict(arm=arm, aspect=aspect, seed=seed, problems=problems, **_git_state()))
                    raise AssertionError(f'{path}: V0c failed twice: {problems}')
                rb_student.write_outputs(path, dict(img=out), dict(meta, sha256=rb_student.sha_array(out),
                                                                   vc3_attempts=len(problems) + 1, vc3_problems=problems))
                arrays, info = rb_student.read_outputs(path)
            assert info['sha256'] == info['arrays']['img']['sha256'] == rb_student.sha_array(out)
            result[aspect, seed] = out
    return result


def held(outputs, seeds, device):
    """Guarded held inference of G's sample heads (rule §4.1 V0a before, V1a before seed 4)."""
    outputs = paths.check_outputs(outputs)
    paths.check_imports()
    gate = read_gate(outputs, 'v0a')
    recorded = gate['parts']['cpu']['items']['g_sample/weights']['sha256']
    device = rg_train.check_device(device)
    torch.set_num_threads(8)
    return infer_set(sample_g(outputs).held_dir, G_LABEL, 'sample', seeds,
                     lambda seed, aspect: sample_weights(outputs, seed, aspect, recorded), held_images(), device)


# ================================================================== V0a
def compare_blocks(ours, theirs):
    """Items of a block against 12-24's K4 block (rule §3.1 scorer check): d, A_cand, each A_e, both bounds within
    1e-6; the entry sets, the outcome kind and every bootstrap d."""
    items = {'d': _close(ours['d'], theirs['d']), 'A_cand': _close(ours['A_cand'], theirs['A_cand']),
             'interval_lo': _close(ours['interval'][0], theirs['interval'][0]),
             'interval_hi': _close(ours['interval'][1], theirs['interval'][1]),
             'entries': {'pass': sorted(ours['A_e']) == sorted(theirs['A_e']), 'ours': sorted(ours['A_e']),
                         'theirs': sorted(theirs['A_e'])}}
    for entry in sorted(theirs['A_e']):
        items[f'A_e/{entry}'] = (_close(ours['A_e'][entry], theirs['A_e'][entry]) if entry in ours['A_e']
                                 else {'pass': False, 'ours': None})
    a, b = np.asarray(ours['bootstrap_d']), np.asarray(theirs['bootstrap_d'])
    items['bootstrap_d'] = {'pass': bool(a.shape == b.shape and np.abs(a - b).max() < TOL), 'n_ours': len(a),
                            'n_theirs': len(b)}
    items['outcome'] = {'pass': ours['outcome_id'].split('-')[-1] == theirs['outcome_id'].split('-')[-1],
                        'ours': ours['outcome_id'], 'theirs': theirs['outcome_id']}
    return items


def scorer_check(scale, cache, n_resamples=2000, workers=8):
    """The scorer on 12-24's saved K4 outputs, K4 in the candidate's place, against its block of `RB_VERDICT[scale]`;
    the candidate's fixed partitions also equal 12-24's saved K4 partitions."""
    theirs_file = RB_VERDICT[scale]
    record = json.loads(theirs_file.read_text())
    assert record['held_sha256'] == HELD_SHA256 and record['scale'] == scale
    theirs = record['blocks']['K4']
    assert len(theirs['bootstrap_d']) == n_resamples == 2000
    ours = _clean(score_scale(k4(scale), scale, Path(cache), n_resamples, workers))
    items = {f'scorer/{scale}/{k}': v for k, v in compare_blocks(ours, theirs).items()}
    equal = {}
    for seed in SEEDS:
        with np.load(paths.RB_PART2 / f'partitions/{scale}/K4/emotion__s{seed}__r20.npz', allow_pickle=False) as z:
            theirs_p = z['labels']
        with np.load(Path(cache) / scale / 'K4' / f'emotion__s{seed}__r20.npz', allow_pickle=False) as z:
            equal[str(seed)] = bool(np.array_equal(z['labels'], theirs_p))
    items[f'scorer/{scale}/partitions_equal'] = {'pass': all(equal.values()), 'equal': equal}
    items[f'scorer/{scale}/sources'] = {'pass': True, 'rb_verdict': str(theirs_file),
                                        'rb_verdict_sha256': sha_file(theirs_file), 'ours_d': ours['d']}
    return items


def g_full_outputs_check():
    """G's full-scale held outputs have the SHA-256 12-27's stage C metadata recorded (`reported.json`, all three
    aspects, five seeds)."""
    recorded = json.loads(G_FULL_RECORD.read_text())['G']['sources']['G_outputs']
    assert sorted(recorded) == sorted(ASPECTS)
    got = {}
    for aspect in ASPECTS:
        assert sorted(recorded[aspect]) == [str(s) for s in SEEDS]
        for seed in SEEDS:
            path = G_FULL_HELD / f'{aspect}__s{seed}.npz'
            ours, theirs = sha_file(path), recorded[aspect][str(seed)]['output_sha256']
            assert recorded[aspect][str(seed)]['path'].endswith(f'/{path.name}')
            got[f'{aspect}__s{seed}'] = dict(ours=ours, recorded=theirs, equal=ours == theirs)
    return {'pass': all(v['equal'] for v in got.values()), 'files': got, 'record': str(G_FULL_RECORD),
            'record_sha256': sha_file(G_FULL_RECORD)}


def g_sample_weights_record():
    """SHA-256 of G's sample weights at seeds 0 to 3 (12-27's files), recorded by V0a."""
    sha = {}
    for seed in SAMPLE_FROZEN_SEEDS:
        folder = paths.RG_TRAIN / paths.G_ARM / f's{seed}'
        record = json.loads((folder / 'run.json').read_text())
        assert record['status'] == 'ok' and record['variant'] == paths.G_ARM and record['seed'] == seed
        for aspect in ASPECTS:
            sha[f's{seed}/{aspect}'] = sha_file(folder / f'{aspect}.pt')
    return {'pass': len(sha) == len(SAMPLE_FROZEN_SEEDS) * len(ASPECTS), 'sha256': sha}


def states_equal_r65_k4():
    """12-25's R6-5 seed-1 best states and 12-24's K4 seed-1 weights hold the same tensors (the files differ in bytes)."""
    equal = {}
    for aspect in ASPECTS:
        equal[aspect] = rg_train.states_equal(R6_5_S1 / f'{aspect}.pt', K4_SAMPLE / f'weights/K4/{aspect}__s1.pt')
    return {'pass': all(equal.values()), 'equal': equal}


def v0a_cpu(outputs, workers=8, n_resamples=2000):
    cache = Path(outputs) / 'stage0/regression/partitions'
    items = {}
    for scale in SCALES:
        items.update(scorer_check(scale, cache, n_resamples, workers))
        print(f'V0a scorer {scale}: ' + ('pass' if all(v['pass'] for k, v in items.items() if f'/{scale}/' in k)
                                          else 'FAIL'), flush=True)
    items['g_full/outputs_sha256'] = g_full_outputs_check()
    items['g_sample/weights'] = g_sample_weights_record()
    items['sample/r6_5_s1_states_equal_k4'] = states_equal_r65_k4()
    return items


def v0a_gpu(device):
    """Held inference from G's full seed-0 weights equals the saved seed-0 outputs; this round's sample inference from
    12-25's R6-5 seed-1 weights equals 12-24's K4 seed-1 outputs, bit for bit, per aspect."""
    device = rg_train.check_device(device)
    torch.set_num_threads(8)
    x = held_images()
    image_sha = rb_student.sha_array(x)
    items = {}
    # 12-25's R6-5 files carry no device; the K4 weights record beside them (same tensors, V0a cpu part) does.
    for key, seed, saved_dir, weights_of, record_of in (
            ('g_full/seed0_inference', 0, G_FULL_HELD, lambda a: G_FULL_WEIGHTS / f'{a}__s0.pt',
             lambda a: G_FULL_WEIGHTS / f'{a}__s0.pt'),
            ('sample/r6_5_s1_inference', 1, K4_SAMPLE / 'held/K4', lambda a: R6_5_S1 / f'{a}.pt',
             lambda a: K4_SAMPLE / f'weights/K4/{a}__s1.pt')):
        for aspect in ASPECTS:
            saved_path = saved_dir / f'{aspect}__s{seed}.npz'
            saved, info = rb_student.read_outputs(saved_path)
            weights = weights_of(aspect)
            recorded = torch.device(weights_record(record_of(aspect)))
            assert torch.device(device) == recorded, 'inference device differs from training device'
            out = infer(weights, aspect, x, recorded)
            bits, same_image = _bits_equal(out, saved['img']), image_sha == info['image_sha256']
            items[f'{key}/{aspect}'] = {'pass': bits and same_image, 'bits_equal': bits,
                                        'image_sha256_equal': same_image, 'weights_sha256': sha_file(weights),
                                        'saved': str(saved_path)}
    return items


V0A_GROUPS = {'cpu': ('scorer/sample/d', 'scorer/full/d', 'scorer/sample/partitions_equal',
                      'scorer/full/partitions_equal', 'g_full/outputs_sha256', 'g_sample/weights',
                      'sample/r6_5_s1_states_equal_k4'),
              'gpu': tuple(f'{k}/{a}' for k in ('g_full/seed0_inference', 'sample/r6_5_s1_inference')
                           for a in ASPECTS)}


def write_v0a(outputs, part, items):
    """Merge one part into gates/v0a.json. The gate passes when both parts are there, each passes with every expected
    item, and both carry the hashes of the code that is on disk now."""
    assert part in V0A_GROUPS
    path = gate_path(outputs, 'v0a')
    record = json.loads(path.read_text()) if path.is_file() else dict(name='v0a', parts={})
    now = code_sha256()
    record['parts'][part] = dict(items=_clean(items), code_sha256=now, code_files=code_files(), **_git_state(),
                                 **{'pass': bool(items) and all(v['pass'] for v in items.values())
                                    and set(V0A_GROUPS[part]) <= set(items)})
    record['code_sha256'] = now
    record['pass'] = (set(record['parts']) == set(V0A_GROUPS)
                      and all(p['pass'] and p['code_sha256'] == now for p in record['parts'].values()))
    _atomic_json(path, record)
    return record


# ================================================================== V0c and V0d
def v0c_items(cand, scale):
    """V0c on a candidate's three aspects and five seeds at one scale: per file, pass or the problem."""
    items = {}
    for aspect in ASPECTS:
        for seed in SEEDS:
            try:
                _, sha = load_candidate(cand, scale, aspect, seed)
                items[f'{aspect}__s{seed}'] = {'pass': True, 'output_sha256': sha}
            except (AssertionError, FileNotFoundError) as error:
                items[f'{aspect}__s{seed}'] = {'pass': False, 'problem': str(error)}
    return items


def check_v0c(outputs):
    items = {scale: v0c_items(cand, scale) for scale, cand in (('sample', sample_g(outputs)), ('full', FULL_G))}
    names, sha = rb_student.held_names()
    passed = sha == HELD_SHA256 and all(i['pass'] for s in items.values() for i in s.values())
    return write_gate(outputs, 'v0c', passed, items, held_sha256=sha, n_rows=N_HELD)


def check_v0d(outputs):
    """Seed policy over the sample set: every output's weights SHA-256 is the current weights' (seeds 0 to 3: the V0a
    record), no failure record, one training of seed 4 (a V1b rerun, rule V1d, is listed), V0c reruns listed."""
    outputs = Path(outputs)
    recorded = read_gate(outputs, 'v0a')['parts']['cpu']['items']['g_sample/weights']['sha256']
    held_dir = sample_g(outputs).held_dir
    items = {}
    for seed in SEEDS:
        for aspect in ASPECTS:
            key = f'{aspect}__s{seed}'
            try:
                assert not failed_path(held_dir, aspect, seed).exists(), 'a failure record exists'
                _, info = rb_student.read_outputs(held_dir / f'{aspect}__s{seed}.npz')
                _, sha = sample_weights(outputs, seed, aspect, recorded)
                assert info['weights_sha256'] == sha, 'the weights changed after the held output was written'
                items[key] = {'pass': info['vc3_attempts'] <= 2, 'weights_sha256': sha,
                              'v0c_attempts': info['vc3_attempts'], 'v0c_problems': info['vc3_problems']}
            except (AssertionError, Refusal, FileNotFoundError) as error:
                items[key] = {'pass': False, 'problem': str(error)}
    folder = paths.run_dir(outputs, S.name(S.config()), 4)
    try:
        run = json.loads((folder / 'run.json').read_text())
        aside = [str(p) for p in rg_train.va3_failures_aside(outputs, S.name(S.config()), 4)]
        items['seed4_trained_once'] = {'pass': run['status'] == 'ok' and len(aside) <= 1 and (len(aside) == 1) == (
            'rerun_of' in run), 'status': run['status'], 'v1b_reruns': aside, 'rerun_of': run.get('rerun_of'),
            'trainings': 1 + len(aside)}
    except (FileNotFoundError, KeyError) as error:
        items['seed4_trained_once'] = {'pass': False, 'problem': repr(error)}
    return write_gate(outputs, 'v0d', all(i['pass'] for i in items.values()), items)


# ================================================================== the verdict
def verdict(outputs, n_resamples=2000, workers=8):
    """stage0/verdict.json: per scale d, A_G, each A_e, bounds, outcome; E0. Needs v0a, v0c and v0d (current code).
    Written once: a second call must give the same record."""
    outputs = paths.check_outputs(outputs)
    paths.check_imports()
    gates = {name: gate_path(outputs, name) for name in ('v0a', 'v0c', 'v0d')}
    for name in gates:
        read_gate(outputs, name)
    cache = outputs / 'stage0/partitions'
    blocks = {scale: _clean(score_scale(cand, scale, cache, n_resamples, workers))
              for scale, cand in (('sample', sample_g(outputs)), ('full', FULL_G))}
    outcomes = {scale: block['outcome_id'] for scale, block in blocks.items()}
    record = dict(schema=1, stage=0, rule_commit=paths.RULE_COMMIT, held_sha256=HELD_SHA256, blocks=blocks,
                  outcomes=outcomes, E0=e0(outcomes), n_resamples=n_resamples,
                  gates={name: sha_file(path) for name, path in gates.items()}, code_sha256=code_sha256(),
                  v0b='pending (ticket 11, the re-derivation from the rule)')
    path = outputs / 'stage0/verdict.json'
    record = _clean(record)
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved == record, f'{path}: refusing to replace a different verdict'
        return saved
    _atomic_json(path, record)
    return record


# ================================================================== CLI
@threadpool_limits.wrap(limits=8)
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outputs', type=Path, default=paths.OUTPUTS)
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('gate', choices=['v0a', 'v0c', 'v0d'])
    c.add_argument('--part', choices=['cpu', 'gpu'], help='v0a only')
    c.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    h = sub.add_parser('held')
    h.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    h.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    sub.add_parser('verdict')
    args = parser.parse_args(argv)
    outputs = paths.check_outputs(args.outputs)
    paths.check_imports()
    try:
        if args.command == 'check':
            if args.gate == 'v0a':
                assert args.part, 'check v0a needs --part cpu or --part gpu'
                started = time.monotonic()
                items = v0a_cpu(outputs, args.workers) if args.part == 'cpu' else v0a_gpu(args.device)
                record = write_v0a(outputs, args.part, items)
                bad = sorted(k for k, v in items.items() if not v['pass'])
                print(f'v0a {args.part}: {"PASS" if record["parts"][args.part]["pass"] else "FAIL"} {bad} '
                      f'({time.monotonic() - started:.0f}s); gate pass={record["pass"]}', flush=True)
                return 0 if record['parts'][args.part]['pass'] else 1
            gate = check_v0c(outputs) if args.gate == 'v0c' else check_v0d(outputs)
            print(f'{args.gate}: {"PASS" if gate["pass"] else "FAIL"} {gate_path(outputs, args.gate)}', flush=True)
            return 0 if gate['pass'] else 1
        if args.command == 'held':
            held(outputs, args.seeds, args.device)
            print(f'stage 0 held: seeds {args.seeds} done in {sample_g(outputs).held_dir}', flush=True)
            return 0
        record = verdict(outputs, workers=args.workers)
        print(f'stage 0: {record["outcomes"]} E0={record["E0"]} {outputs / "stage0/verdict.json"}', flush=True)
        return 0
    except Refusal as error:
        raise SystemExit(f'REFUSED: {error}')


if __name__ == '__main__':
    raise SystemExit(main())
