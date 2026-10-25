"""Gates, candidate lists, deferral and the end estimate for the queue scripts of the encoders-and-genre round.

The queues (`eg_queue_*.sh`, ticket 16) call this file between steps. Reads JSON files and decides; trains and scores
nothing itself, except `replace`, `score3` and `lift-score`, which call the scorers after their gate. Exit 0 (go) or
exit 1 with one `EG_STOP <reason>` line. Commands (rule §2.1, §4.1 to §4.3; spec §3.5, D15):
  gate NAME [--cand K] [--stage N] [--phase P]
      x0 | v5 --cand K | x1 | x2 | stage3 --cand K | stage4 --cand K | v2 --stage N | vl1 | vl2 | vg1 |
      vg2 --phase first|all|fresh | parity (R6-5 fresh-seed weights equal 12-24's K4; before any C outcome)
  candidates STAGE     3: K6 K7 K8 K5 minus deferred minus those with a block in verdict_stage3.json;
                       4: those with an S1 outcome other than S1-NO, minus deferred, minus blocks in verdict_stage4.json
  check-verdict STAGE  every listed candidate has a block (RB's check)
  a28                  'models|scales of B1/B6 needed|{model: scale}' for stage 5
  defer                print the deferred candidates (part2/deferral.json)
  defer --estimate-end T   T = HH:MM or ISO time of the estimated end: past the deadline (default next 11:00, D15)
                       records the next candidate of K5, K8, K7, K6 without a stage 3 block as deferred
  defer --release K    a deferred candidate runs now (the record keeps the history; never dropped)
  budget               one line: time estimate per remaining step, end, route (12-24's measured times, scaled by width)
  replace --stage N --cand K --reason R   replace a kept no-outcome block after a fix: calls eg_score --replace; when a
                       replaced stage 4 block moves the candidate's stage 5 scale, its stage 5 block is replaced too
  score3 first|all     eg_score3, only after the vg1 gate; lift-score: eg_lift score, only after the vl1 gate
Not here: eg_train3 `--rerun-failed` (at most once; eg_train3 refuses a second one itself, VG4).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import eg_paths
from eg_encoders import BY_STORE, BY_STUDENT, CANDIDATES, DEFERRAL_ORDER, ENCODERS

ROUND = eg_paths.ROUND
FULL, A28 = 'artelingo_full', 'artelingo28'     # eg_v5's dataset folder names (keys of v5.json `stores`)
CALIBRATION = ('clip_b32', 'siglip2_so400m')    # X0's two stores
X_STAGE = {'x1': ('sample', 3), 'x2': ('full', 4)}
X_PARTS = ('student', 'matched', 'scorer')
PHASE_SOURCE = {'first': 'score_first.json', 'all': 'score_all.json', 'fresh': 'score_fresh.json'}


class Stop(Exception):
    """A gate that does not hold; the message is the status line."""


def need(cond, message):
    if not cond:
        raise Stop(message)


def load(path):
    path = Path(path)
    need(path.is_file(), f'missing {path}')
    try:
        rec = json.loads(path.read_text())
    except ValueError as err:
        raise Stop(f'{path} is not JSON: {err}') from err
    need(isinstance(rec, dict), f'{path} is not a JSON object')
    return rec


def cand_of(name):
    need(name in CANDIDATES, f'--cand must be one of {CANDIDATES}, got {name}')
    return BY_STUDENT[name]


def outcome_of(block):
    """A verdict block's outcome ID; None when the block has no outcome (a V check failed). Key `outcome_id` (RB's)."""
    assert 'outcome_id' in block, f'verdict block without an outcome_id key: {sorted(block)[:8]}'
    return block['outcome_id']


def blocks_of(out, stage, must=True):
    path = Path(out) / f'part2/verdict_stage{stage}.json'
    if not path.exists() and not must:
        return {}
    rec = load(path)
    need(isinstance(rec.get('blocks'), dict), f'{path.name} has no blocks')
    return rec['blocks']


# --- gates ---------------------------------------------------------------------------------------------------------

def gate_x0(out):
    """X0 (rule §4.1): the changed and the old extractor agree on 512 images of both calibration stores."""
    rec = load(out / 'stores/x0.json')
    need(set(CALIBRATION) <= set(rec.get('stores', {})), f'X0: stores/x0.json lacks {CALIBRATION}')
    bad = [s for s in CALIBRATION if rec['stores'][s].get('equal') is not True]
    need(rec.get('pass') is True and not bad, f'X0 failed ({bad or "pass false"}): no new store is written, stage i stops')
    return 'X0 passed'


def v5_entry(out, enc, dataset):
    rec = load(out / 'stores/v5.json')
    entry = rec.get('stores', {}).get(f'{dataset}/{enc.store}')
    need(entry is not None, f'V5: no check of {dataset}/{enc.store} in stores/v5.json')
    need(isinstance(entry.get('pass'), bool), f'V5: {dataset}/{enc.store} has no boolean pass')
    return entry


def gate_v5(out, cand, a28=False):
    """V5 for the encoder of `cand`. The ArtELingo store decides stage 3; the ArtELingo-28 store alone only blocks the
    A outcome (rule §4.1), so it is only required with `a28`."""
    enc = cand_of(cand)
    full = v5_entry(out, enc, FULL)
    need(full['pass'], f'V5: the {FULL}/{enc.store} store of {cand} failed; {cand} has no outcome (reported with the cause)')
    small = v5_entry(out, enc, A28)
    if a28:
        need(small['pass'], f'V5: the {A28}/{enc.store} store of {cand} failed; {cand} has no A outcome')
    return f'V5 {cand}: {FULL} pass' + ('' if small['pass'] else f'; {A28} FAILED (no A outcome for {cand})')


def regression(out, name):
    """X1 or X2: the student, matched and scorer halves each pass (rule §4.1)."""
    scale = X_STAGE[name][0]
    for part in X_PARTS:
        path = out / f'part2/regression/{name}_{part}.json'
        rec = load(path)
        items = rec.get('items')
        need(isinstance(items, dict) and items, f'{name.upper()}: {path.name} has no items')
        bad = sorted(k for k, v in items.items() if v.get('pass') is not True)
        need(rec.get('pass') is True and not bad, f'{name.upper()}: {path.name} fails ({bad[:4] or "pass false"})')
        need(rec.get('scale', scale) == scale, f'{name.upper()}: {path.name} is at scale {rec.get("scale")}, not {scale}')
    return f'{name.upper()}: student, matched and scorer halves pass'


def gate_stage3(out, cand):
    return '; '.join((gate_x0(out), gate_v5(out, cand), regression(out, 'x1')))


def s1_outcome(out, cand):
    blocks = blocks_of(out, 3)
    need(cand in blocks, f'stage 4 of {cand}: no stage 3 block in verdict_stage3.json (S1 outcome not written)')
    return outcome_of(blocks[cand])


def gate_stage4(out, cand):
    cand_of(cand)
    msg = regression(out, 'x2')
    s1 = s1_outcome(out, cand)
    need(s1 is not None, f'stage 4 of {cand}: no S1 outcome (stage 3 block has none)')
    need(s1 != 'S1-NO', f'stage 4 of {cand}: S1-NO, no stage 4')
    return f'{msg}; S1 outcome {s1}'


def newer(path, source, what):
    need(path.stat().st_mtime >= source.stat().st_mtime, f'{path.name} is older than {source.name}: rerun {what}')


def gate_v2(out, stage):
    """V2 agrees, is not older than the stage's verdict file, and re-derived every block that has an outcome."""
    need(stage in (3, 4, 5), 'v2 --stage must be 3, 4 or 5')
    path, vpath = out / f'part2/rederive_v2_stage{stage}.json', out / f'part2/verdict_stage{stage}.json'
    rec = load(path)
    need(rec.get('agree') is True, f'V2 does not agree for stage {stage}')
    blocks = blocks_of(out, stage)
    newer(path, vpath, 'V2')
    missing = [c for c, b in blocks.items() if outcome_of(b) is not None and c not in rec.get('candidates', {})]
    need(not missing, f'V2 stage {stage} lacks {missing}: rerun V2')
    return f'V2 agrees for stage {stage}'


def gate_vl1(out):
    """VL1: (b), (c), VL3 (ticket 07's vl1.json) and (a) (ticket 08's vl1_scorer.json) both pass."""
    for name in ('vl1.json', 'vl1_scorer.json'):
        rec = load(out / 'stage_ii' / name)
        need(rec.get('pass') is True, f'VL1: stage_ii/{name} does not pass; stage ii stops (rule §4.2)')
        if 'p_sha256' in rec:
            need(rec['p_sha256'] == eg_paths.P_SHA256, f'VL1: stage_ii/{name} has another P SHA-256')
    return 'VL1 holds: vl1.json and vl1_scorer.json pass'


def gate_vl2(out):
    path = out / 'stage_ii/rederive_vl2.json'
    rec = load(path)
    need(rec.get('agree') is True, 'VL2 does not agree: no stage ii outcome is stated')
    newer(path, out / 'stage_ii/lift.json' if (out / 'stage_ii/lift.json').exists() else path, 'VL2')
    need((out / 'stage_ii/lift.json').is_file(), 'VL2: stage_ii/lift.json is missing')
    return 'VL2 agrees'


def gate_vg1(out):
    for name in ('vg1.json', 'vg1_scorer.json'):
        rec = load(out / 'stage_iii' / name)
        need(rec.get('pass') is True, f'VG1: stage_iii/{name} does not pass; no variant is trained (rule §4.3)')
        if 'p_sha256' in rec:
            need(rec['p_sha256'] == eg_paths.P_SHA256, f'VG1: stage_iii/{name} has another P SHA-256')
    return 'VG1 holds: vg1.json and vg1_scorer.json pass'


def gate_vg2(out, phase):
    need(phase in PHASE_SOURCE, f'vg2 --phase must be one of {sorted(PHASE_SOURCE)}')
    path = out / f'stage_iii/rederive_vg2_{phase}.json'
    rec = load(path)
    need(rec.get('phase', phase) == phase, f'VG2: {path.name} is for phase {rec.get("phase")}')
    need(rec.get('agree') is True, f'VG2 ({phase}) does not agree: no stage iii outcome is stated')
    source = out / 'stage_iii' / PHASE_SOURCE[phase]
    need(source.is_file(), f'VG2: {source.name} is missing')
    newer(path, source, f'VG2 {phase}')
    return f'VG2 agrees ({phase})'


def gate_parity(out):
    rec = load(out / 'stage_iii/fresh_parity.json')
    seeds = rec.get('seeds', {})
    need(set(seeds) >= {'1', '2', '3'}, 'parity: fresh_parity.json lacks seeds 1 to 3')
    bad = sorted(s for s, v in seeds.items() if v.get('pass') is not True)
    need(rec.get('pass') is True and not bad, f'parity: seeds {bad or "(pass false)"} differ from 12-24\'s K4: no C outcome')
    return 'R6-5 fresh-seed weights equal 12-24\'s K4'


def run_gate(out, args):
    name = args.arg
    if name == 'x0':
        return gate_x0(out)
    if name == 'v5':
        return gate_v5(out, args.cand)
    if name in ('x1', 'x2'):
        return regression(out, name)
    if name == 'stage3':
        return gate_stage3(out, args.cand)
    if name == 'stage4':
        return gate_stage4(out, args.cand)
    if name == 'v2':
        return gate_v2(out, args.stage)
    if name == 'vl1':
        return gate_vl1(out)
    if name == 'vl2':
        return gate_vl2(out)
    if name == 'vg1':
        return gate_vg1(out)
    if name == 'vg2':
        return gate_vg2(out, args.phase)
    if name == 'parity':
        return gate_parity(out)
    raise Stop('gate names: x0 v5 x1 x2 stage3 stage4 v2 vl1 vl2 vg1 vg2 parity')


# --- deferral and candidates ---------------------------------------------------------------------------------------

def deferral_path(out):
    return Path(out) / 'part2/deferral.json'


def read_deferral(out):
    path = deferral_path(out)
    if not path.exists():
        return dict(schema=1, deferred=[], history=[])
    rec = json.loads(path.read_text())
    assert set(rec['deferred']) <= set(CANDIDATES) and isinstance(rec['history'], list), f'{path}: bad content'
    return rec


def deferred(out):
    return list(read_deferral(out)['deferred'])


def candidates(out, stage, todo=True):
    """Stage 3: K6 K7 K8 K5 minus deferred. Stage 4: those with an S1 outcome other than S1-NO. `todo`: also minus
    those that already have a block in that stage's verdict file (a no-outcome block is replaced with `replace`)."""
    names = [c for c in CANDIDATES if c not in deferred(out)]
    if stage == 4:
        s3 = blocks_of(out, 3, must=False)
        names = [c for c in names if c in s3 and outcome_of(s3[c]) not in (None, 'S1-NO')]
    elif stage != 3:
        raise Stop('candidates STAGE: 3 or 4')
    if todo:
        done = blocks_of(out, stage, must=False)
        names = [c for c in names if c not in done]
    return names


def check_verdict(out, stage, cands):
    blocks = blocks_of(out, stage)
    missing = [c for c in cands if c not in blocks]
    need(not missing, f'verdict_stage{stage}.json has no block for {missing}')
    noout = [c for c in cands if outcome_of(blocks[c]) is None]
    return f'verdict stage {stage}: ' + ', '.join(f'{c}={outcome_of(blocks[c])}' for c in cands) + \
        (f' (no outcome: {noout}; reported with the cause)' if noout else '')


def x2_failed(out):
    """X2 failed: a half exists and does not pass (rule §4.1: stage 5 then runs at the sample scale)."""
    for part in X_PARTS:
        path = out / f'part2/regression/x2_{part}.json'
        if path.exists() and json.loads(path.read_text()).get('pass') is not True:
            return True
    return False


def a28_scale(out, model, s4=None):
    """Rule §2.1: full when the model's stage 4 block has an outcome, else sample; sample for all when X2 failed."""
    if x2_failed(out):
        return 'sample'
    s4 = blocks_of(out, 4, must=False) if s4 is None else s4
    return 'full' if model in s4 and outcome_of(s4[model]) is not None else 'sample'


def a28_plan(out):
    """Stage 5 models: the candidates not deferred that have a stage 3 block (outputs exist) and whose ArtELingo-28
    store passed V5; their scales, and the scales of B1 / B6 outputs they need."""
    s3 = blocks_of(out, 3, must=False)
    models = [c for c in CANDIDATES if c not in deferred(out) and c in s3]
    models = [m for m in models if v5_entry(out, BY_STUDENT[m], A28)['pass']] if models else models
    s4 = blocks_of(out, 4, must=False)
    scales = {m: a28_scale(out, m, s4) for m in models}
    return models, scales


def to_time(text, now):
    """HH:MM -> the next such time after `now`; or an ISO time."""
    if 'T' in text or '-' in text:
        t = datetime.fromisoformat(text)
        return t if t.tzinfo else t.replace(tzinfo=now.tzinfo)
    hh, mm = (int(x) for x in text.split(':'))
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def now_of(text):
    return datetime.fromisoformat(text).replace(tzinfo=ZoneInfo('Europe/Amsterdam')) if text else \
        datetime.now(ZoneInfo('Europe/Amsterdam'))


def defer(out, estimate_end, deadline, release, now):
    """Spec D15, rule §2.1: decided from the end estimate before that candidate's stage 3; order K5, K8, K7, K6; a
    deferred candidate is run later, never dropped. One candidate per call: estimate again (`budget`) and call again."""
    assert not str(Path(out).resolve()).startswith(str(eg_paths.RB_OUT)), 'never write into 12-24\'s outputs'
    rec = read_deferral(out)
    if release:
        need(release in rec['deferred'], f'{release} is not deferred')
        rec['deferred'].remove(release)
        rec['history'].append(dict(candidate=release, action='release', at=now.isoformat(timespec='minutes')))
        msg = f'released {release}'
    elif estimate_end is None:
        return f'deferred: {" ".join(rec["deferred"]) or "none"}'
    else:
        end, limit = to_time(estimate_end, now), to_time(deadline, now)
        if end <= limit:
            return f'no deferral: estimated end {end:%F %H:%M} is not after the deadline {limit:%F %H:%M}'
        s3 = blocks_of(out, 3, must=False)
        pick = [c for c in DEFERRAL_ORDER if c not in rec['deferred'] and c not in s3]
        need(pick, f'estimated end {end:%F %H:%M} is past {limit:%F %H:%M} and no candidate is left to defer '
                   f'(deferred {rec["deferred"]}; the others have a stage 3 block)')
        cand = pick[0]
        rec['deferred'].append(cand)
        rec['history'].append(dict(candidate=cand, action='defer', estimate_end=end.isoformat(timespec='minutes'),
                                   deadline=limit.isoformat(timespec='minutes'), decided_at=now.isoformat(timespec='minutes'),
                                   remaining=[c for c in CANDIDATES if c not in rec['deferred']]))
        msg = f'deferred {cand} (end {end:%F %H:%M} past {limit:%F %H:%M}); run `budget` again'
    path = deferral_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(rec, indent=1) + '\n')
    tmp.replace(path)
    return msg + f'; deferred now: {" ".join(rec["deferred"]) or "none"}'


# --- end estimate --------------------------------------------------------------------------------------------------
# Minutes (low, high). Measured in 12-24 (RB/20261025_recipe_backbones_run_log.md, 2026-10-25 07:24 to 08:51), on
# K2 (width 1152), K3 (512), K4 (2048): training time is almost flat in width, so a line through K3 and K4 is used.
# Low = that line; high = low x HIGH (the unmeasured width scaling and a queue wait). E marks an estimate without a
# 12-24 measurement.
HIGH = 1.4


def line(w, a, b):
    """Seconds at width w from the measured (512: a, 2048: b)."""
    return a + (w - 512) * (b - a) / (2048 - 512)


def seconds_per_step(w):
    return dict(
        stores_full=10 * 60 * w / 512,       # M CLIP B/32 store 10 to 12 min (S); DINOv2 1024: 20 to 25 (linear in width)
        stores_a28=5 * 60 * w / 512,         # S 5 to 8 min
        x0=4 * 60,                           # E two stores, 512 images, two extractors
        v5=3 * 60,                           # E per store (retrieval check)
        train3=line(w, 179, 192), infer3=15, matched3=line(w, 14, 16),      # M stage 3 sample, seed 0 to 4 per candidate
        train4=line(w, 499, 515), infer4=15, matched4=line(w, 28, 32),      # M stage 4 full
        score3=441 / 3, v2_3=375 / 3, score4=433 / 3, v2_4=376 / 3,         # M per candidate
        score5=539 / 4, v2_5=51 / 4, infer5=3 * 7,                          # M per model; B1, B6 and the model
        report=3613 / 3,                                                    # M report queue, per candidate and stage
    )


FIXED_SECONDS = dict(
    x1=65 + 10 + 122,       # M stage 2 of 12-24: student, matched, scorer check
    x2=178 + 13 + 115,      # M stage 3a
    stage_ii_infer=7 * 3 * 40,   # E seven encoders, three aspects, P (29,232 rows is 2.4 x the held set)
    stage_ii_score=600,     # E lifts and intervals (12-24's part 1 bootstrap about 7 min)
    stage_iii_train=7 * 101 + 6 * 101,   # M part 1 recipes 53 to 101 s each; 7 variants and 6 fresh-seed runs, upper end
    stage_iii_score=2 * 370 + 370,       # M 12-24's pick 370 s; first, all, fresh
)


def budget(out, now):
    out = Path(out)
    cands = candidates(out, 3, todo=False)
    s3, s4 = blocks_of(out, 3, must=False), blocks_of(out, 4, must=False)
    s5 = blocks_of(out, 5, must=False)
    todo3 = [c for c in cands if c not in s3]
    todo4 = [c for c in cands if c not in s4 and (c in todo3 or (c in s3 and outcome_of(s3[c]) not in (None, 'S1-NO')))]
    exists = lambda rel: (out / rel).exists()
    v5s = load(out / 'stores/v5.json').get('stores', {}) if exists('stores/v5.json') else {}
    steps = []

    def add(label, sec, skip=False, n=1):
        lo = 0 if skip else sec * n / 60
        steps.append((label, lo, lo * HIGH))

    add('x0', seconds_per_step(512)['x0'], exists('stores/x0.json'))
    for c in todo3:
        w = BY_STUDENT[c].width
        s = seconds_per_step(w)
        store = BY_STUDENT[c].store
        add(f'{c}:store', s['stores_full'], f'{FULL}/{store}' in v5s)
        add(f'{c}:store28', s['stores_a28'], f'{A28}/{store}' in v5s)
        add(f'{c}:v5', 2 * s['v5'], f'{FULL}/{store}' in v5s and f'{A28}/{store}' in v5s)
    add('x1', FIXED_SECONDS['x1'], all(exists(f'part2/regression/x1_{p}.json') for p in X_PARTS))
    for c in todo3:
        s = seconds_per_step(BY_STUDENT[c].width)
        add(f'{c}:stage3', s['train3'] + s['infer3'] + s['matched3'] + s['score3'] + s['v2_3'])
    add('x2', FIXED_SECONDS['x2'], all(exists(f'part2/regression/x2_{p}.json') for p in X_PARTS))
    for c in todo4:
        s = seconds_per_step(BY_STUDENT[c].width)
        add(f'{c}:stage4', s['train4'] + s['infer4'] + s['matched4'] + s['score4'] + s['v2_4'])
    models = [c for c in cands if c not in s5]
    for m in models:
        s = seconds_per_step(BY_STUDENT[m].width)
        add(f'{m}:stage5', s['infer5'] + s['score5'] + s['v2_5'])
    add('stage_ii', FIXED_SECONDS['stage_ii_infer'] + FIXED_SECONDS['stage_ii_score'], exists('stage_ii/lift.json'))
    add('stage_iii_train', FIXED_SECONDS['stage_iii_train'], exists('stage_iii/score_all.json'))
    add('stage_iii_score', FIXED_SECONDS['stage_iii_score'], exists('stage_iii/score_all.json'))
    reports = sum(seconds_per_step(BY_STUDENT[c].width)['report'] for c in cands) * 2 / 60   # stages 3 and 4, CPU beside
    lo, hi = sum(x[1] for x in steps), sum(x[2] for x in steps)
    deadline = now.replace(hour=11, minute=0, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    end_lo, end_hi = now + timedelta(minutes=lo), now + timedelta(minutes=hi)
    if end_hi <= deadline:
        route = 'local, the high end is before the deadline'
    else:
        route = (f'high end past {deadline:%H:%M}: `eg_run.py defer --estimate-end {end_hi:%H:%M}` defers the next of '
                 f'{" ".join(DEFERRAL_ORDER)}; else Monday 19:00 (D15)')
    parts = ' '.join(f'{label}={a:.0f}-{b:.0f}' for label, a, b in steps if b > 0) or 'nothing left'
    return (f'EG_BUDGET now={now:%F %H:%M} candidates={" ".join(cands) or "none"} (deferred {" ".join(deferred(out)) or "none"}) '
            f'steps_min[{parts}] chain={lo:.0f}-{hi:.0f}min end={end_lo:%F %H:%M}..{end_hi:%F %H:%M} '
            f'deadline={deadline:%F %H:%M} reports_cpu_beside={reports:.0f}min route: {route}')


# --- calls to the scorers ------------------------------------------------------------------------------------------

def call(cmd, print_only):
    print('EG_CALL ' + ' '.join(str(x) for x in cmd), flush=True)
    if print_only:
        return 0
    return subprocess.run([str(x) for x in cmd], cwd=ROUND).returncode


def py(script, *args):
    return [sys.executable, ROUND / script, *args]


def replace(out, stage, cand, reason, print_only, workers):
    """After a fix, replace a kept no-outcome block (ticket 05 review). A replaced stage 4 block that moves the
    candidate's stage 5 scale replaces its stage 5 block too (same reason)."""
    need(reason and reason.strip(), 'replace needs a non-empty V-check reason (--reason)')
    need(stage in (3, 4), 'replace --stage must be 3 or 4')
    cand_of(cand)
    block = blocks_of(out, stage).get(cand)
    need(block is not None, f'verdict_stage{stage}.json has no block for {cand}: nothing to replace')
    need(outcome_of(block) is None, f'the {stage} block of {cand} has an outcome ({outcome_of(block)}); it is kept')
    old_scale = a28_scale(out, cand)
    base = ['--outputs', out, '--workers', workers, '--candidates', cand, '--replace', reason]
    code = call(py('eg_score.py', '--stage', stage, *base), print_only)
    if stage == 4:
        new_scale = a28_scale(out, cand)
        s5 = blocks_of(out, 5, must=False).get(cand)
        if print_only or (s5 is not None and (new_scale != s5.get('scale') or old_scale != new_scale)):
            why = f'{reason} (stage 4 block of {cand} replaced; stage 5 scale {old_scale} to {new_scale})'
            if s5 is not None or print_only:
                code = call(py('eg_score.py', '--stage', 5, '--outputs', out, '--workers', workers,
                               '--candidates', cand, '--replace', why), print_only) or code
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['gate', 'candidates', 'check-verdict', 'a28', 'defer', 'budget', 'replace',
                                        'score3', 'lift-score'])
    ap.add_argument('arg', nargs='?')
    ap.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    ap.add_argument('--cand')
    ap.add_argument('--stage', type=int)
    ap.add_argument('--phase')
    ap.add_argument('--candidates', nargs='*')
    ap.add_argument('--estimate-end')
    ap.add_argument('--deadline', default='11:00')
    ap.add_argument('--release')
    ap.add_argument('--reason')
    ap.add_argument('--now', help='ISO time for tests')
    ap.add_argument('--print-only', action='store_true', help='replace, score3, lift-score: print the call only')
    ap.add_argument('--workers', type=int, default=8)
    args, extra = ap.parse_known_args(argv)
    out = args.outputs
    try:
        cmd = args.command
        if extra and cmd not in ('score3', 'lift-score'):
            ap.error(f'unknown arguments {extra}')
        if cmd == 'gate':
            print(f'EG_GATE {args.arg} ok: {run_gate(out, args)}')
        elif cmd == 'candidates':
            print(' '.join(candidates(out, int(args.arg))))
        elif cmd == 'check-verdict':
            print('EG_VERDICT ' + check_verdict(out, int(args.arg), args.candidates or candidates(out, int(args.arg), todo=False)))
        elif cmd == 'a28':
            models, scales = a28_plan(out)
            print(' '.join(models) + '|' + ' '.join(sorted(set(scales.values()))) + '|' + json.dumps(scales))
        elif cmd == 'defer':
            print('EG_DEFER ' + defer(out, args.estimate_end, args.deadline, args.release, now_of(args.now)))
        elif cmd == 'budget':
            print(budget(out, now_of(args.now)))
        elif cmd == 'replace':
            return replace(out, args.stage, args.cand, args.reason, args.print_only, args.workers)
        elif cmd == 'score3':
            need(args.arg in ('first', 'all'), 'score3 first|all')
            if args.arg == 'all':
                gate_vg2_free = None  # noqa: F841  (score all needs only VG1 and the first scoring; eg_score3 checks the rest)
            print(f'EG_GATE vg1 ok: {gate_vg1(out)}')
            return call(py('eg_score3.py', args.arg, '--outputs', out, '--workers', args.workers, *extra), args.print_only)
        elif cmd == 'lift-score':
            print(f'EG_GATE vl1 ok: {gate_vl1(out)}')
            return call(py('eg_lift.py', 'score', '--outputs', out, '--workers', args.workers, *extra), args.print_only)
    except Stop as err:
        print(f'EG_STOP {args.command} {args.arg or ""}: {err}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
