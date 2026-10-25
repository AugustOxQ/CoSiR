"""Stage driver of the recipe-and-backbones round: gates, candidate lists and the end estimate for the queue scripts.

The queues (`rb_queue_*.sh`) call this file between steps. Every command prints one line and exits 0 (go) or 1 (stop;
the line says why). It reads files only; it trains and scores nothing. Commands:
  gate NAME            stage1 | vp1 | part2 | stage3 | stage4 | stage5   (spec §3.5, rule §4.1, §4.2)
  vp3 --recipe R       VP3 (rule §4.1): refuse a recipe outside the plan or a finished recipe without --forced
  cause LOG            the VP2 cause from a failed rb_part1 log (exit 1 when the failure is not a VP2 assertion)
  recipes              the six recipes in rule order
  pick                 the pick line (only after VP1 agrees)
  candidates STAGE     K2 K3 [K4] [K5] for stage 3; for stage 4 those with an S1 outcome other than S1-NO
  check-verdict STAGE  every candidate has a block in verdict_stage<STAGE>.json
  triggers             K4 / K5 from pick.json
  a28                  'models|scales of B1/B6 outputs|model: scale' for stage 5
  budget               end estimate against the next 11:00 (spec D32)
Env: RB_OUTPUTS (outputs folder), RB_DEFER (e.g. "K5": candidates deferred for time, logged, never dropped).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import rb_paths
import rb_rows

CANDIDATES = ('K2', 'K3', 'K4', 'K5')
# Expected regression item names (rule §4.2 V1, ticket 16): student/ from rb_student.regression_items, matched/ from
# rb_matched.regress, scorer/ from rb_score.compare_with_ct and run_check. Variable names need at least one.
EXPECTED = {
    'student/': ['student/emotion', 'student/style', 'student/genre', 'student/held_list'],
    'matched/': ['matched/qwen_s0/eval_prob', 'matched/qwen_s0/held_prob', 'matched/qwen_s0/mapper_state',
                 'matched/qwen_s0/config', 'matched/ct_mapper_infer/P3_s0', 'matched/ct_mapper_infer/P2_s0'],
    'scorer/': ['scorer/A_K1', 'scorer/entries', 'scorer/d', 'scorer/interval_lo', 'scorer/interval_hi',
                'scorer/bootstrap_d', 'scorer/outcome'],
}
AT_LEAST_ONE = ('scorer/A_e/', 'scorer/partitions/')


class Stop(Exception):
    """A gate that does not hold; the message is the status line."""


def load(path):
    path = Path(path)
    if not path.exists():
        raise Stop(f'missing {path}')
    return json.loads(path.read_text())


def need(cond, message):
    if not cond:
        raise Stop(message)


def outputs_of(args):
    out = Path(args.outputs)
    if args.smoke:
        need(str(out.resolve()).startswith('/tmp/claude-0/'), f'smoke needs outputs under /tmp/claude-0/: {out}')
    return out


# --- gates ---------------------------------------------------------------------------------------------------------

def gate_stage1(out, smoke):
    """V0: stage 1 needs a passed stage0.json (smoke: a smoke record whose pass is false by design)."""
    rec = load(out / 'stage0.json')
    if smoke:
        need(rec.get('smoke') is True, 'stage0.json is not a smoke record')
        return 'stage 0 smoke record present (pass not required in smoke)'
    need(rec.get('smoke') is False and rec.get('pass') is True, 'V0: stage0.json has not passed; part 1 does not run')
    return 'stage 0 passed'


def gate_vp1(out, smoke):
    """No pick is stated before VP1 agrees (rule §4.1): rederive_vp1.json agree true, written after pick.json."""
    pick = out / 'part1/pick.json'
    need(pick.exists(), f'no {pick}')
    rec = load(out / 'part1/rederive_vp1.json')
    need(rec.get('agree') is True, 'VP1 does not agree: trace and fix, no pick is stated (rule §4.1)')
    need((out / 'part1/rederive_vp1.json').stat().st_mtime >= pick.stat().st_mtime,
         'rederive_vp1.json is older than pick.json: rerun VP1')
    return 'VP1 agrees'


def gate_part2(out, smoke):
    gate_stage1(out, smoke)
    return gate_vp1(out, smoke)


def regression_complete(out, name):
    """Stage 2 / 3a: every expected item present and passing, the file's own pass true."""
    rec = load(out / f'part2/regression/{name}.json')
    items = rec.get('items', {})
    missing = [k for names in EXPECTED.values() for k in names if k not in items]
    missing += [g for g in AT_LEAST_ONE if not any(k.startswith(g) for k in items)]
    need(not missing, f'V1: regression {name} lacks items {missing[:4]}{"..." if len(missing) > 4 else ""}')
    bad = sorted(k for k, v in items.items() if v.get('pass') is not True)
    need(rec.get('pass') is True and not bad, f'V1: regression {name} has failing items {bad[:4]}')
    return f'regression {name}: {len(items)} items pass'


def s1_outcomes(out):
    return load(out / 'part2/verdict_stage3.json')['blocks']


def gate_stage3(out, smoke):
    gate_part2(out, smoke)
    return regression_complete(out, 'stage2')


def gate_stage4(out, smoke):
    gate_part2(out, smoke)
    msg = regression_complete(out, 'stage3a')
    v2 = load(out / 'part2/rederive_v2_stage3.json')
    need(v2.get('agree') is True, 'V2 does not agree for stage 3: stage 4 does not start')
    return msg


def gate_stage5(out, smoke):
    gate_part2(out, smoke)
    return regression_complete(out, 'stage3a')


def gate_v2(stage):
    def gate(out, smoke):
        rec = load(out / f'part2/rederive_v2_stage{stage}.json')
        need(rec.get('agree') is True, f'V2 does not agree for stage {stage}: the reported readings do not run')
        return f'V2 agrees for stage {stage}'
    return gate


GATES = {'stage1': gate_stage1, 'vp1': gate_vp1, 'part2': gate_part2, 'stage3': gate_stage3, 'stage4': gate_stage4,
         'stage3a': gate_stage5, 'stage5': gate_stage5, 'v2_3': gate_v2(3), 'v2_4': gate_v2(4)}


def vp3(out, recipe, forced):
    """VP3 (rule §4.1): after any export on P exists no recipe is added, dropped or retrained, except a rerun forced by a
    failed V check. The plan is the six recipes, fixed at the first call; a finished recipe (P exports of all three
    aspects) is not trained again without --forced; every call is logged."""
    need(recipe in rb_rows.ORDER, f'VP3: {recipe} is not one of the six recipes')
    part1 = out / 'part1'
    part1.mkdir(parents=True, exist_ok=True)
    plan = part1 / 'plan.json'
    if plan.exists():
        need(json.loads(plan.read_text()) == list(rb_rows.ORDER), 'VP3: the recipe plan changed')
    else:
        plan.write_text(json.dumps(list(rb_rows.ORDER)))
    done = all(list((part1 / 'train' / recipe).glob(f'{a}__s0__P.npz')) for a in rb_rows_aspects())
    need(not done or forced, f'VP3: {recipe} is finished (P exports exist); retraining needs --forced')
    with open(part1 / 'vp3_log.jsonl', 'a') as f:
        f.write(json.dumps(dict(recipe=recipe, forced=forced, finished_before=done, time=stamp())) + '\n')
    return f'VP3 ok for {recipe}' + (' (forced rerun, logged)' if forced else '')


def rb_rows_aspects():
    return rb_paths.RB_ASPECTS


def stamp():
    return datetime.now(ZoneInfo('Europe/Amsterdam')).isoformat(timespec='seconds')


def vp2_cause(log):
    """The last line of a failed rb_part1 log when it is a VP2 assertion of `check_p`; else Stop."""
    lines = [x.strip() for x in Path(log).read_text().splitlines() if x.strip()]
    need(lines, f'empty log {log}')
    last = lines[-1]
    vp2 = last.startswith('AssertionError') and any(t in last for t in ('nonfinite rows', 'norm error', 'float16'))
    need(vp2, f'not a VP2 assertion: {last[:160]}')
    return last.replace('\n', ' ')[:300]


# --- candidates ----------------------------------------------------------------------------------------------------

def triggers(out):
    pick = load(out / 'part1/pick.json')
    t = pick['triggers']
    need(set(t) == {'K4', 'K5'}, f'pick.json triggers {t}')
    return t


def deferred():
    return [c for c in os.environ.get('RB_DEFER', '').split() if c]


def candidates(out, stage, todo=True):
    """Triggered candidates in the order K2, K3, K4, K5, minus RB_DEFER. Stage 4: only those with an S1 outcome
    other than S1-NO (rule §4.2); a candidate without an outcome has no stage 4. `todo`: also minus those that already
    have a block in this stage's verdict file, so a rerun after a deferral trains and scores only the new ones
    (rb_score merges blocks by name and refuses to recompute a block whose code commit changed)."""
    t = triggers(out)
    names = [c for c in CANDIDATES if c in ('K2', 'K3') or t[c]]
    for c in deferred():
        need(c in CANDIDATES, f'RB_DEFER names {c}')
    names = [c for c in names if c not in deferred()]
    if stage == 4:
        blocks = s1_outcomes(out)
        names = [c for c in names if blocks.get(c, {}).get('outcome_id') not in (None, 'S1-NO')]
    if todo:
        verdict = out / f'part2/verdict_stage{stage}.json'
        scored = set(json.loads(verdict.read_text())['blocks']) if verdict.exists() else set()
        names = [c for c in names if c not in scored]
    return names


def check_verdict(out, stage, cands):
    blocks = load(out / f'part2/verdict_stage{stage}.json')['blocks']
    missing = [c for c in cands if c not in blocks]
    need(not missing, f'verdict_stage{stage}.json has no block for {missing}')
    noout = [c for c in cands if blocks[c].get('outcome_id') is None]
    return f'verdict stage {stage}: ' + ', '.join(f'{c}={blocks[c].get("outcome_id")}' for c in cands) + \
        (f' (no outcome: {noout}; reported with the cause)' if noout else '')


def a28_plan(out):
    """Models of stage 5 (rb_a28's default list minus RB_DEFER) and the scales whose B1 / B6 outputs they need.
    "Stage 4 ran" means "ran to an outcome" (a block with outcome_id not None in verdict_stage4.json)."""
    models = ['K1', 'K2', 'K3'] + [c for c in ('K4', 'K5') if triggers(out)[c]]
    models = [m for m in models if m not in deferred()]
    v4 = out / 'part2/verdict_stage4.json'
    blocks = json.loads(v4.read_text())['blocks'] if v4.exists() else {}
    scales = {m: ('full' if m == 'K1' or blocks.get(m, {}).get('outcome_id') is not None else 'sample') for m in models}
    return models, scales


# --- end estimate (spec §9 budget, D32) ----------------------------------------------------------------------------
# (low, high) minutes. M = measured (gpu_checks notes, build 16), S = spec §9 line, E = estimate of this ticket.
GPU_STEPS = dict(
    stage0=(4, 8, 'S regressions about 10 min in all'),
    part1=(20, 30, 'S part 1 about 25 min'),
    stores_clip=(10, 12, 'S CLIP store about 10 min'),
    stores_a28=(5, 8, 'S ArtELingo-28 about 5 min'),
    stores_dino=(20, 25, 'S DINOv2 about 35 min with its student; store part is E'),
    reg2=(5, 8, 'S regressions about 10 min for the two'),
    reg3a=(5, 8, 'S'),
    k23_sample=(3, 5, 'S 45 min for K2 and K3 at both scales; a quarter at sample (per candidate)'),
    k23_full=(8, 12, 'S three quarters at full (per candidate)'),
    k4_sample=(4, 5, 'S K4 about 15 min at both scales'),
    k4_full=(11, 15, 'S'),
    k5_sample=(4, 5, 'S DINOv2 about 35 min; student part'),
    k5_full=(11, 15, 'S'),
    stage5=(5, 10, 'E K1, B1 and B6 inference on ArtELingo-28'),
)
CPU_STEPS = dict(
    pick=(7, 10, 'M part 1 bootstrap about 7 min at 8 workers'),
    vp1=(7, 10, 'M about 7 min'),
    check2=(7, 12, 'E scorer check, 2,000 draws'),
    check3a=(7, 12, 'E'),
    verdict=(7, 12, 'E 2,000 draws per stage'),
    v2=(7, 15, 'E'),
    report=(30, 40, 'E per candidate per stage at 8 workers (ticket 13)'),
    a28_verdict=(5, 10, 'E'),
)


def budget(out, now=None):
    """End estimate: the foreground chain in order (GPU steps and the CPU steps between them), plus the two part 2
    report queues, which start after their stage's V2, run one after the other (CPU lock) and beside the GPU chain.
    End = the later of the chain's end and the stage 4 report's end. Steps whose output file exists cost 0. The
    numbers are in GPU_STEPS and CPU_STEPS (`M` measured, `S` spec §9, `E` estimate); CPU load from the reports slows
    the chain's own CPU steps, which this does not model."""
    now = now or datetime.now(ZoneInfo('Europe/Amsterdam'))
    try:
        t = triggers(out) if (out / 'part1/pick.json').exists() else {'K4': True, 'K5': True}
    except Stop:
        t = {'K4': True, 'K5': True}
    cands = [c for c in CANDIDATES if (c in ('K2', 'K3') or t[c]) and c not in deferred()]
    done = lambda rel: (out / rel).exists()
    chain, marks = [], {}

    def add(label, table, name, count=1, skip=False):
        lo, hi, src = table[name]
        chain.append((label, 0 if skip else lo * count, 0 if skip else hi * count, src))
        marks[label] = (sum(c[1] for c in chain), sum(c[2] for c in chain))

    n23 = sum(c in ('K2', 'K3') for c in cands)
    add('stage0', GPU_STEPS, 'stage0', skip=done('stage0.json'))
    add('part1', GPU_STEPS, 'part1', skip=done('part1/pick.json'))
    add('pick', CPU_STEPS, 'pick', skip=done('part1/pick.json'))
    add('vp1', CPU_STEPS, 'vp1', skip=done('part1/rederive_vp1.json'))
    store = lambda rel: (Path(rb_paths.PRE) / rel).exists()
    add('stores_clip', GPU_STEPS, 'stores_clip', skip=store('artelingo_full/clip_b32/index.json'))
    add('stores_a28', GPU_STEPS, 'stores_a28', skip=all(store(f'artelingo28/qwen3vl_emb_2b_aspect_prompts/{k}/done.json')
                                                        for k in ('emotion_focus', 'style_word', 'genre_word')))
    if 'K5' in cands:
        add('stores_dino', GPU_STEPS, 'stores_dino', skip=store('artelingo_full/dinov2_large/index.json') and
            store('artelingo28/dinov2_large/index.json'))
    add('reg2', GPU_STEPS, 'reg2', skip=done('part2/regression/stage2.json'))
    add('check2', CPU_STEPS, 'check2', skip=done('part2/regression/stage2.json'))
    v3 = done('part2/verdict_stage3.json')
    add('k23_sample', GPU_STEPS, 'k23_sample', n23, skip=v3)
    for k in ('K4', 'K5'):
        if k in cands:
            add(f'{k.lower()}_sample', GPU_STEPS, f'{k.lower()}_sample', skip=v3)
    add('verdict3', CPU_STEPS, 'verdict', skip=v3)
    add('v2_3', CPU_STEPS, 'v2', skip=done('part2/rederive_v2_stage3.json'))
    add('reg3a', GPU_STEPS, 'reg3a', skip=done('part2/regression/stage3a.json'))
    add('check3a', CPU_STEPS, 'check3a', skip=done('part2/regression/stage3a.json'))
    v4 = done('part2/verdict_stage4.json')
    add('k23_full', GPU_STEPS, 'k23_full', n23, skip=v4)
    for k in ('K4', 'K5'):
        if k in cands:
            add(f'{k.lower()}_full', GPU_STEPS, f'{k.lower()}_full', skip=v4)
    add('verdict4', CPU_STEPS, 'verdict', skip=v4)
    add('v2_4', CPU_STEPS, 'v2', skip=done('part2/rederive_v2_stage4.json'))
    add('stage5', GPU_STEPS, 'stage5')
    add('a28_verdict', CPU_STEPS, 'a28_verdict')
    add('v2_5', CPU_STEPS, 'v2')
    n = len(cands)
    ends = []
    for i in (0, 1):
        rep = CPU_STEPS['report'][i] * n
        start3 = marks['v2_3'][i]
        end3 = start3 + (0 if done('part2/reported_stage3.json') else rep)
        start4 = max(marks['v2_4'][i], end3)
        end4 = start4 + (0 if done('part2/reported_stage4.json') else rep)
        ends.append(max(marks['v2_5'][i], end4))
    deadline = now.replace(hour=11, minute=0, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    lo_t, hi_t = now + timedelta(minutes=ends[0]), now + timedelta(minutes=ends[1])
    if hi_t <= deadline:
        route = 'local; the high end is before 11:00'
    else:
        route = ('the high end is past 11:00: the run chat splits the run (spec D32) and records why; first RB_DEFER="K5", '
                 'then "K4 K5" (deferred, never dropped)' + ('' if lo_t <= deadline else '; the low end too'))
    return dict(start=now.isoformat(timespec='minutes'), candidates=cands,
                chain_minutes=dict(low=marks['v2_5'][0], high=marks['v2_5'][1]),
                steps=[dict(step=a, low=b, high=c, source=d) for a, b, c, d in chain],
                estimated_end_low=lo_t.isoformat(timespec='minutes'),
                estimated_end_high=hi_t.isoformat(timespec='minutes'),
                deadline=deadline.isoformat(timespec='minutes'), route=route)


# --- command line --------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['gate', 'vp3', 'cause', 'recipes', 'pick', 'candidates', 'check-verdict',
                                        'triggers', 'a28', 'budget'])
    ap.add_argument('arg', nargs='?')
    ap.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--recipe')
    ap.add_argument('--forced', action='store_true')
    ap.add_argument('--candidates', nargs='*')
    args = ap.parse_args(argv)
    try:
        out = outputs_of(args)
        if args.command == 'gate':
            need(args.arg in GATES, f'gate names: {sorted(GATES)}')
            print(f'RB_GATE {args.arg} ok: {GATES[args.arg](out, args.smoke)}')
        elif args.command == 'vp3':
            print('RB_GATE vp3 ' + vp3(out, args.recipe, args.forced))
        elif args.command == 'cause':
            print(vp2_cause(args.arg))
        elif args.command == 'recipes':
            print(' '.join(rb_rows.ORDER))
        elif args.command == 'pick':
            gate_vp1(out, args.smoke)
            pick = load(out / 'part1/pick.json')
            print(f"RB_PICK outcome={pick['outcome']} recipe={pick['recipe']} triggers={json.dumps(pick['triggers'])} "
                  f'(VP1 agrees)')
        elif args.command == 'candidates':
            print(' '.join(candidates(out, int(args.arg))))
        elif args.command == 'check-verdict':
            print('RB_VERDICT ' + check_verdict(out, int(args.arg),
                                                args.candidates or candidates(out, int(args.arg), todo=False)))
        elif args.command == 'triggers':
            print(' '.join(c for c, v in triggers(out).items() if v))
        elif args.command == 'a28':
            models, scales = a28_plan(out)
            print(' '.join(models) + '|' + ' '.join(sorted(set(scales.values()))) + '|' + json.dumps(scales))
        elif args.command == 'budget':
            print(json.dumps(budget(out), indent=1))
    except Stop as err:
        print(f'RB_STOP {args.command} {args.arg or ""}: {err}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
