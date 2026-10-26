"""Stage A scorer, seed 0 (rule §3.1, §4.1, §4.3 "Weights"): each arm's P exports against R6-5, the guard, the climb, the
seed-0 winner v*_A, the A outcome, the weights lists, and the VA1 scorer half.

Modes (`python rg_score.py MODE`):
  climb      the ladder rungs run so far (GR-8, GR-16, GR-32, ...): the next rung or none (rule §4.1 "The climb"),
             appended to stage_a/climb.json; refuses before GR-16 and GR-32 exist
  a          every arm run (the climb's rungs, GN-8, GS-0.02, GS-0.01, GQ-8): guard, v*_A, A-NEW / A-SAME, levels
             against T_g, kept weights and pending deletions -> stage_a/verdict.json (VA4: any other arm set is refused)
  check va1  the scorer gives 12-25's score_all.json `results` of R6-5 and GR-8 and T_g within 1e-6
             -> regression/va1_scorer.json
  check vb1  the frozen fresh-seed exports of GR-8 and R6-5 hash to 12-25's score_fresh.json and the fresh-seed scorer
             gives its `effects` within 1e-6 -> stage_b/vb1.json
  report     the reported readings of stages A and B (spec §3.6; rg_report.py) -> reported_ab.{json,md}
  b          stage B (rule §3.1, §4.2): GR-2, GR-4, GR-8 and v*_A (when A-NEW) on seeds 1 to 3 against R6-5: effects, C
             levels, v_held, B-PICK / B-NONE -> stage_b/verdict.json (refuses without a passing vb1.json and stage A's
             verdict; VB4: any other arm or seed is refused)

Statistic: 12-25's (`eg_score3`: the RB rule §3.1 with R6-5 in R0-5's place). The functions that do not call
`eg_variants` are imported from it; the ones that do, or that write under its `stage_iii/`, are copied here with a
comment naming the original. Decisions use unrounded values. Caches go under `<outputs>/stage_a/partitions/<arm>/s0/`;
the frozen 12-24 and 12-25 files are only read.
"""
import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

import numpy as np

import rg_paths
import rg_variants as V
import eg_score3 as E
import rb_ami
import rb_data
import rb_pick

ASPECTS = rg_paths.ASPECTS
BASE = V.BASE
SEED = 0
RANGES = V.RANGES
PROMISING = V.T_G - RANGES['genre']             # T_g - r_genre, unrounded
TOL = V.TOL                                      # VA1

# imported from eg_score3 (they do not touch eg_variants or stage_iii paths)
is_guarded = E.is_guarded                        # the guard, rule §4.1: both bounds inclusive
condition_record = E.condition_record
failed_record = E.failed_record
aspect_effects = E.aspect_effects


# ------------------------------------------------------------------ the rule's conditions (pure)
def climb_decision(records):
    """The climb's next rung (rule §4.1 "The climb"). `records` = {rung: record} of the ladder rungs run so far, GR-8
    first, contiguous in `rg_variants.LADDER` order, with GR-16 and GR-32 among them. The newest rung's successor (double
    the newest) runs when the newest is guarded and its unrounded A_genre is strictly above that of every other guarded
    rung run (GR-8 included); it stops when the newest is failed, unguarded, tied or lower, and after GR-256.
    -> dict(newest, next (rung or None), reason, still_rising, rivals {rung: A_genre of every other guarded rung})."""
    run = list(records)
    assert 'GR-8' in records, 'the climb needs GR-8'
    assert 'GR-16' in records, 'the climb needs GR-16 first'
    assert 'GR-32' in records, 'the climb needs GR-32 first'
    assert run == list(V.LADDER[:len(run)]), f'ladder rungs run {run} are not contiguous from GR-8 in {V.LADDER} order'
    newest = run[-1]
    rec = records[newest]
    rivals = {r: records[r]['A_genre'] for r in run[:-1] if records[r]['guarded']}
    best = max(rivals.values(), default=-np.inf)
    if rec['status'] != 'ok' or rec['A_genre'] is None:
        reason, rising = 'failed', False
    elif not rec['guarded']:
        reason, rising = 'unguarded', False
    elif rec['A_genre'] > best:
        reason, rising = 'rises', True
    else:
        reason, rising = ('tied' if rec['A_genre'] == best else 'lower'), False
    if newest == V.LADDER[-1]:                # nothing after GR-256; a rung still rising there is reported
        return dict(newest=newest, next=None, reason='cap', still_rising=bool(rising), rivals=rivals)
    return dict(newest=newest, next=V.LADDER[len(run)] if rising else None, reason=reason, still_rising=bool(rising),
                rivals=rivals)


def decide_a(results, eq_pass, rungs):
    """The A verdict from every arm's record: VA4 on the arm set, v*_A, the outcome, each arm's level, the weights lists
    (arms in `A_TIE_ORDER`), and which arm stage B trains on seeds 1 to 3 beside GR-2 and GR-4."""
    check_va4(list(results), rungs)
    ordered = {n: results[n] for n in V.A_TIE_ORDER if n in results}
    winner = pick_winner(ordered, eq_pass)
    kept, rest = split_weights(ordered)
    outcome = a_outcome(winner)
    return dict(winner=winner, A_genre_winner=ordered[winner]['A_genre'], outcome=outcome, eq_pass=bool(eq_pass),
                levels={n: level(r['A_genre']) for n, r in ordered.items()}, kept_weights=kept, pending_deletions=rest,
                trains_fresh=[winner] if outcome == 'A-NEW' else [])


def climb_entry(records):
    """One decision of `stage_a/climb.json`: `climb_decision` plus every rung's A_genre, e_emotion, e_style, guard."""
    d = climb_decision(records)
    rungs = {}
    for n, r in records.items():
        eff = r['effects']
        rungs[n] = dict(status=r['status'], cause=r['cause'], A_genre=r['A_genre'], e_genre=r['e_genre'],
                        e_emotion=eff['emotion']['e'] if eff else None, e_style=eff['style']['e'] if eff else None,
                        guarded=r['guarded'], **({'rerun_of': r['rerun_of']} if 'rerun_of' in r else {}))
    return dict(rungs_run=list(records), rungs=rungs, **d)


def _same_numbers(a, b):
    """Two climb entries agree on every rung's status and numbers (within 1e-9)."""
    if a['rungs_run'] != b['rungs_run'] or (a['next'], a['reason']) != (b['next'], b['reason']):
        return False
    for n, x in a['rungs'].items():
        y = b['rungs'][n]
        if (x['status'], x['guarded']) != (y['status'], y['guarded']):
            return False
        for k in ('A_genre', 'e_genre', 'e_emotion', 'e_style'):
            if (x[k] is None) != (y[k] is None) or (x[k] is not None and abs(x[k] - y[k]) > 1e-9):
                return False
    return True


def extend_climb_log(log, entry):
    """The log after this decision (a new list). Decisions come in order: the first is on GR-8, GR-16, GR-32; each next one
    adds exactly the rung the one before named; nothing follows a stop; the same rungs again must carry the same numbers
    and append nothing."""
    log = list(log)
    if not log:
        assert entry['rungs_run'] == list(V.LADDER[:3]), \
            f'the first climb decision is on {list(V.LADDER[:3])} (GR-16 and GR-32 first); got {entry["rungs_run"]}'
        return [entry]
    last = log[-1]
    if entry['rungs_run'] == last['rungs_run']:
        assert _same_numbers(last, entry), f'climb.json: the decision on {entry["rungs_run"]} differs from the recorded one'
        return log
    assert last['next'] is not None, f'the climb stopped after {last["newest"]} ({last["reason"]}); {entry["rungs_run"]} was run anyway'
    assert entry['rungs_run'] == last['rungs_run'] + [last['next']], \
        f'the decision after {last["newest"]} comes before {last["next"]}\'s: rungs run {entry["rungs_run"]}'
    return log + [entry]


def climb_rungs(log):
    """The ladder rungs of a finished climb (the last decision's `rungs_run`, GR-8 first); refuses an empty or an
    unfinished log."""
    assert log, 'no climb decision recorded (run `climb` first)'
    assert log[-1]['next'] is None, f'the climb is not finished: the next rung is {log[-1]["next"]}'
    return list(log[-1]['rungs_run'])


def pick_winner(results, eq_pass):
    """v*_A (rule §4.1): among GR-8 and every new arm run, the guarded one with the largest unrounded A_genre; on equal
    values GR-8 first, then `rg_variants.A_TIE_ORDER`. GQ-8 is in the pool only when EQ passed. None if no arm is
    guarded (GR-8 is, per 12-25's score_all.json, so a winner always exists)."""
    assert set(results) <= set(V.A_TIE_ORDER), sorted(set(results) - set(V.A_TIE_ORDER))
    best = None
    for n in V.A_TIE_ORDER:
        r = results.get(n)
        if r is None or r['status'] != 'ok' or not r['guarded'] or (n == 'GQ-8' and not eq_pass):
            continue
        if best is None or r['A_genre'] > results[best]['A_genre']:
            best = n
    return best


def a_outcome(winner):
    """A-SAME when v*_A is GR-8 (stage B runs GR-2 and GR-4 only); A-NEW otherwise (v*_A trains on seeds 1 to 3 too)."""
    assert winner is not None, 'no winner'
    return 'A-SAME' if winner == 'GR-8' else 'A-NEW'


def level(a_genre):
    """An arm's level against T_g (reported, decides nothing): target, promising, neither; None for a failed run."""
    if a_genre is None:
        return None
    if a_genre >= V.T_G:
        return 'target'
    return 'promising' if a_genre >= PROMISING else 'neither'


def split_weights(results):
    """(kept, rest) in the order of `results` (rule §4.3 "Weights"): A_genre >= T_g - r_genre keeps its weights; the rest,
    failed runs included, go on the pending deletions list for the wrap-up."""
    kept = [v for v, r in results.items() if r['A_genre'] is not None and r['A_genre'] >= PROMISING]
    return kept, [v for v in results if v not in kept]


def check_va4(arms, climb_rungs):
    """VA4: the scored arm set is the climb's record (its ladder rungs, GR-8 first) plus `rg_variants.A_OTHER`."""
    want = sorted(list(climb_rungs) + list(V.A_OTHER))
    assert sorted(arms) == want, f'VA4: arm set {sorted(arms)} is not the climb\'s record plus A_OTHER: {want}'


# ------------------------------------------------------------------ stage B's conditions (pure; rule §4.2)
FRESH = V.FRESH_SEEDS                            # (1, 2, 3)
FROZEN_ARMS = ('GR-8', BASE)                     # their fresh-seed runs are 12-25's
# eg_score3's c_outcome and genre_vs_targets read eg_variants' RANGES and T_G: the same numbers as this round's
assert E.V.RANGES == V.RANGES and E.V.T_G == V.T_G and tuple(E.FRESH) == tuple(FRESH)
c_level = E.c_outcome                             # C-HOLDS / C-QUAL / C-FAILS (EG rule §4.3)


def fresh_effect(labels, parts, arm, ref, n_draws, workers=8, caches=None):
    """One aspect's fresh-seed result of `arm` against `ref`: {A, Abar, e, lo, hi} (`eg_score3.fresh_aspect`). `parts` =
    {name: {seed: int [3, n]}} for the two names, the same seeds. Each call draws I_b then S_b from its own
    `default_rng(0)`, so every arm of an aspect gets the same draws, which R6-5 shares (rule §3.1)."""
    seeds = tuple(sorted(parts[arm]))
    return E.fresh_aspect(labels, parts, arm, ref, seeds, n_draws, workers, caches)


def b_record(effects, arm):
    """A scored arm's stage B record from `effects` = {aspect: fresh_aspect result}: Ā_genre, ē_genre, its C level and
    whether it is confirmed (C-HOLDS: rule §4.2)."""
    c = c_level(effects)
    abar = effects['genre']['Abar'][arm]
    return dict(status='ok', cause=None, effects=effects, Abar_genre=abar, e_genre=effects['genre']['e'], c_outcome=c,
                confirmed=c == 'C-HOLDS', genre_vs_targets=E.genre_vs_targets(abar))


def failed_b_record(cause):
    """A fresh-seed run that stopped on a non-finite loss or still fails VB3 after its rerun: not confirmed, no level
    (rule §4.2 "Failed runs")."""
    assert cause, 'a failed run needs its cause'
    return dict(status='failed', cause=str(cause), effects=None, Abar_genre=None, e_genre=None, c_outcome=None,
                confirmed=False, genre_vs_targets=None)


def b_arms(verdict_a):
    """Stage B's arms (rule §2.2), in tie order: GR-8, GR-2, GR-4, then v*_A when stage A said A-NEW."""
    assert verdict_a['outcome'] in ('A-SAME', 'A-NEW'), verdict_a['outcome']
    if verdict_a['outcome'] == 'A-SAME':
        assert verdict_a['winner'] == 'GR-8', verdict_a['winner']
        return tuple(V.B_TIE_ORDER)
    assert verdict_a['winner'] in V.A_TIE_ORDER and verdict_a['winner'] != 'GR-8', verdict_a['winner']
    return tuple(V.B_TIE_ORDER) + (verdict_a['winner'],)


def check_vb4(arms, v_star_a, seeds):
    """VB4: no run beyond GR-2, GR-4 and v*_A (when A-NEW) on seeds 1 to 3; GR-8 and R6-5 are the frozen 12-25 runs."""
    want = list(V.B_TIE_ORDER) + ([v_star_a] if v_star_a else [])
    assert sorted(arms) == sorted(want), f'VB4: arm set {sorted(arms)} is not {sorted(want)}'
    assert tuple(seeds) == tuple(FRESH), f'VB4: seeds {tuple(seeds)} are not {tuple(FRESH)}'


def check_fresh_dirs(outputs, allowed):
    """VB4: a run folder `train/<arm>/s<seed>` is stage B's only for an arm in `allowed` on a seed of 1 to 3; other arms
    have seed 0 (stage A's, VA4 checks those) and nothing else. GR-2 and GR-4 have no seed 0 (they are stage B's only), and
    a folder not named `s<int>` is refused."""
    root = Path(outputs) / 'train'
    bad = []
    for path in sorted(root.glob('*/s*')) if root.exists() else []:
        arm, found = path.parent.name, re.fullmatch(r's(\d+)', path.name)
        seed = int(found.group(1)) if found else None
        stage_a = seed == SEED and arm not in V.B_NEW
        if not stage_a and (seed not in FRESH or arm not in set(allowed)):
            bad.append(f'{arm}/{path.name}')
    assert not bad, f'VB4: run folders beyond {sorted(allowed)} on seeds {FRESH}: {bad}'


def decide_b(results, v_star_a):
    """The B verdict from every arm's record (`b_record` / `failed_b_record`): the confirmed arms, v_held (the largest
    unrounded Ā_genre among them; equal values: GR-8, GR-2, GR-4, then v*_A), B-PICK / B-NONE, each arm's C level, and
    the confirmed arms whose Ā_genre is within r_genre of v_held's."""
    order = list(V.B_TIE_ORDER) + ([v_star_a] if v_star_a else [])
    assert sorted(results) == sorted(order), f'VB4: arms {sorted(results)} are not {sorted(order)}'
    confirmed = [a for a in order if results[a]['confirmed']]
    held = None
    for a in confirmed:
        if held is None or results[a]['Abar_genre'] > results[held]['Abar_genre']:
            held = a
    near = [a for a in confirmed if a != held and results[held]['Abar_genre'] - results[a]['Abar_genre'] <= RANGES['genre']]
    return dict(outcome='B-PICK' if held else 'B-NONE', v_held=held, v_held_switches=V.switches(held) if held else None,
                confirmed=confirmed, within_one_range=near, levels={a: results[a]['c_outcome'] for a in order})


def check_frozen_hashes(got, ref):
    """VB1, hashes: the frozen fresh-seed exports of GR-8 and R6-5 hash to 12-25's `score_fresh.json`
    `identity.img_sha256` per arm, seed and aspect. Both are {arm: {seed (str): {aspect: sha256}}}."""
    assert sorted(got) == sorted(ref), f'VB1: arms {sorted(got)} are not {sorted(ref)}'
    for arm in ref:
        assert sorted(got[arm]) == sorted(ref[arm]), f'VB1: {arm} seeds {sorted(got[arm])} are not {sorted(ref[arm])}'
        for seed in ref[arm]:
            assert sorted(got[arm][seed]) == sorted(ref[arm][seed]), f'VB1: {arm} s{seed} aspects differ'
            for aspect, sha in ref[arm][seed].items():
                assert got[arm][seed][aspect] == sha, f'VB1: {arm} s{seed} {aspect}: export hash differs from 12-25\'s'


def compare_effects(got, ref, aspects, smoke):
    """VB1, numbers: {item: {value, expected, diff, pass}} for each aspect's Ā of both arms and ē, and (not in a smoke) both
    bounds, against `ref` = 12-25's `score_fresh.json` `effects`; each within 1e-6."""
    items = {}

    def item(name, value, want):
        items[name] = dict(value=float(value), expected=float(want), diff=float(abs(value - want)),
                           **{'pass': bool(abs(value - want) <= TOL)})

    for a in aspects:
        assert sorted(got[a]['Abar']) == sorted(ref[a]['Abar']), (a, sorted(got[a]['Abar']))
        for arm in ref[a]['Abar']:
            item(f'{a}_Abar_{arm}', got[a]['Abar'][arm], ref[a]['Abar'][arm])
        for k in ('e',) + (() if smoke else ('lo', 'hi')):
            item(f'{a}_{k}', got[a][k], ref[a][k])
    return items


def arm_run_status(runs):
    """('ok' | 'failed', cause) of one arm from its runs {seed: record of `read_fresh_run`}: any failed seed fails the arm."""
    causes = [f'seed {s}: {r["cause"]}' for s, r in sorted(runs.items()) if r['status'] == 'failed']
    return ('failed', '; '.join(causes)) if causes else ('ok', None)


# ------------------------------------------------------------------ files
_json_dump = E._json_dump
_file_sha256 = E._file_sha256


def scorer_code_sha256():
    """Hash this scorer and its transitive top-level imports in the round, EG and RB folders (copied from
    `eg_score3.scorer_code_sha256`, which starts from eg_score3 and looks in the EG and RB folders only). Names as well as
    contents are hashed; lazy imports inside functions are excluded."""
    import hashlib
    folders = (rg_paths.ROUND, Path(E.__file__).parent, Path(rb_ami.__file__).parent)
    pending = [rg_paths.ROUND / 'rg_score.py']
    files = set()
    while pending:
        path = pending.pop()
        if path in files:
            continue
        files.add(path)
        for node in ast.parse(path.read_bytes(), filename=str(path)).body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                for folder in folders:
                    dependency = folder / f'{name}.py'
                    if dependency.is_file():
                        pending.append(dependency)
                        break
    manifest = {f'{p.parent.name}/{p.name}': _file_sha256(p) for p in files}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def code_state():
    """git HEAD, round-local Python changes and the scorer/dependency hash."""
    def git(*a):
        return subprocess.run(['git', '-C', str(rg_paths.ROUND), *a], capture_output=True, text=True, check=True).stdout
    return dict(code_commit=git('rev-parse', 'HEAD').strip(), code_dirty=bool(git('status', '--porcelain', '--', '*.py').strip()),
                code_sha256=scorer_code_sha256())


def stage_dir(outputs):
    return Path(outputs) / 'stage_a'


def cache_dir(outputs, arm, seed=SEED):
    return stage_dir(outputs) / 'partitions' / arm / f's{seed}'


def arm_folder(outputs, arm):
    """GR-8's seed 0 is 12-25's frozen run; every new arm is this round's."""
    return rg_paths.frozen_run('GR-8', SEED) if arm == 'GR-8' else rg_paths.run_dir(outputs, arm, SEED)


def fixed_partitions(outputs, arm, aspect, img, p_sha, seed=SEED):
    """int64 [3, n]: the arm's fixed partitions of one aspect (RB rule §3.1), computed once and cached under
    stage_a/partitions/ (copied from `eg_score3.fixed_partitions`, whose cache sits under stage_iii/)."""
    path = cache_dir(outputs, arm, seed) / f'{aspect}__partitions.npz'
    identity = E._identity(arm, aspect, p_sha, rb_pick.array_sha256(img), seed)
    parts = E._read_npz_with_record(path, 'labels', identity)
    if parts is None:
        parts = rb_ami.partitions(img)
        E._write_npz_with_record(path, 'labels', parts, dict(identity=identity, versions=rb_ami.versions(),
                                                             k=rb_ami.K, states=list(rb_ami.STATES)))
    assert parts.shape == (3, len(img)) and parts.dtype == np.int64
    return parts


class DiskBoots:
    """Per-arm bootstrap scores of one aspect at seed 0, cached under the arm's folder in stage_a/partitions/ (copied from
    `eg_score3.DiskBoots`, whose cache sits under stage_iii/)."""

    def __init__(self, outputs, aspect, n_draws, p_sha, label_sha, img_sha, seed=SEED):
        self.outputs, self.aspect, self.n_draws, self.seed = outputs, aspect, n_draws, seed
        self.p_sha, self.label_sha, self.img_sha = p_sha, label_sha, img_sha

    def _where(self, name):
        path = cache_dir(self.outputs, name, self.seed) / f'{self.aspect}__boot.npz'
        return path, E._identity(name, self.aspect, self.p_sha, self.img_sha[name], self.seed,
                                 label_sha256=self.label_sha, n_draws=self.n_draws)

    def get(self, name):
        path, identity = self._where(name)
        return E._read_npz_with_record(path, 'boot', identity)

    def put(self, name, boot):
        path, identity = self._where(name)
        E._write_npz_with_record(path, 'boot', np.asarray(boot, dtype=np.float64), dict(identity=identity))


# ------------------------------------------------------------------ inputs
def check_run_code(outputs, run, smoke):
    va1 = read_training_va1(outputs, smoke)
    hashes = va1.get('rg_code_sha256')
    assert isinstance(hashes, dict) and set(hashes) == {'rg_train.py', 'rg_loss.py', 'rg_variants.py'}, 'VA1: missing rg_code_sha256'
    assert run.get('rg_code_sha256') == hashes, 'trained by other code than VA1 checked; rerun VA1 or retrain'


def read_run(outputs, arm, p_sha, declared_failed=None, smoke=False):
    """{status, cause, best_epoch, run_sha256, img {aspect: array}} of one arm at seed 0 (copied from `eg_score3.read_run`,
    which reads 12-25's stage iii folders and `eg_variants`' switches). A run.json with status 'failed' has no exports
    read. An arm in `declared_failed` (VA3 still failing after its rerun) has its exports read and must fail VA3 on at
    least one aspect; it is then a failed run with the declared cause. The switches are the arm's rg form (a new arm) or
    12-25's {m, lam, w} form (GR-8 only: its run is 12-25's frozen one). A failed run's `failure_kind` is 'nonfinite_loss' or
    'va3' (this round's trainer; 12-25's spelling is 'vg3'), and a 'va3' failure must carry `rerun_of` (rule §4.1: one rerun
    first). `rerun_of`, when present, goes into the record (VA4's report lists the reruns)."""
    declared_failed = declared_failed or {}
    folder = arm_folder(outputs, arm)
    run_path = folder / 'run.json'
    assert run_path.exists(), f'{arm}: {run_path} missing (the run has not finished)'
    run = json.loads(run_path.read_text())
    if arm != 'GR-8':
        check_run_code(outputs, run, smoke)
    assert run['variant'] == arm and run['seed'] == SEED, (arm, run.get('variant'), run.get('seed'))
    want = V.eg_form(V.ARMS[arm]) if arm == 'GR-8' else V.switches(arm)
    assert run['switches'] == want, f'{arm}: run.json switches {run["switches"]}, expected {want}'
    assert run['status'] in ('ok', 'failed'), (arm, run['status'])
    assert p_sha == rg_paths.P_SHA256, 'P differs from 12-25\'s'
    assert run.get('p_sha256') == p_sha, f'{arm}: run.json p_sha256 {run.get("p_sha256")!r} is not 12-25\'s P'
    assert smoke or not run.get('smoke'), f'{arm}: a smoke run is not scored outside a smoke'
    rec = dict(status=run['status'], cause=run.get('cause'), best_epoch=run.get('best_epoch'),
               run_sha256=_file_sha256(run_path), img=None, **({'rerun_of': run['rerun_of']} if 'rerun_of' in run else {}))
    if run['status'] == 'failed':
        assert run.get('cause'), f'{arm}: failed run without a cause'
        # copied from `eg_score3.read_run` (spelled 'va3' here, 'vg3' there); rule §4.1: a run fails if its loss goes
        # non-finite, or if it still fails VA3 after its (one) rerun
        assert run.get('failure_kind') in ('nonfinite_loss', 'va3'), (arm, 'failure_kind', run.get('failure_kind'))
        assert run['failure_kind'] != 'va3' or 'rerun_of' in run, f'{arm} s0: failed VA3 and has not been rerun; rerun it once first'
        return rec
    img, problems = {}, {}
    for a in ASPECTS:
        with np.load(folder / f'{a}__P.npz', allow_pickle=False) as z:
            assert 'img' in z.files, (arm, a, z.files)
            img[a] = z['img']
        problem = rb_pick.vp2_problem(img[a])                     # VA3: 29,232 finite unit-norm rows
        if problem is not None:
            problems[a] = problem
    if arm in declared_failed:
        assert problems, f'VA3 {arm}: --failed declared it, but its exports pass VA3 on every aspect (rule §4.1)'
        return dict(rec, status='failed', cause=declared_failed[arm], vg3_problem=problems)
    assert not problems, f'VA3 {arm}: {problems} (rerun it, or declare the failure with --failed)'
    return dict(rec, img=img)


def check_arm_dirs(outputs, allowed):
    """VA4: only allowed seed-0 arms; replaced runs must be the current run's forced VA3 origin."""
    root = Path(outputs) / 'train'
    found = sorted(p.parent.name for p in root.glob(f'*/s{SEED}')) if root.exists() else []
    extra = [v for v in found if v not in set(allowed)]
    assert not extra, f'VA4: seed-0 run folders of arms outside the climb\'s record and A_OTHER: {extra}'
    for origin in sorted((Path(outputs) / 'replaced').glob('*/s0*')):
        arm = origin.parent.name
        path = origin / 'run.json'
        assert arm in allowed and path.is_file(), f'VA4: unapproved replaced run {origin}'
        old = json.loads(path.read_text())
        assert (old.get('variant'), old.get('seed'), old.get('status'), old.get('failure_kind')) == (arm, 0, 'failed', 'va3'), f'VA4: replaced run was not a VA3 failure: {origin}'
        current_path = root / arm / 's0' / 'run.json'
        assert current_path.is_file(), f'VA4: no current run for replaced run {origin}'
        current = json.loads(current_path.read_text())
        link = current.get('rerun_of')
        assert link and Path(link).resolve() == origin.resolve(), f'VA4: replaced run is not named by the current rerun_of: {origin}'


def score_arms(outputs, arms, n_draws, workers, declared_failed=None, ctx=None, smoke=False):
    """{'base': {aspect: A}, 'results': {arm: record}, 'identity': {...}} for `arms` against R6-5 (seed 0). `n_draws` None:
    point estimates only (the climb; the records carry A and e but no interval)."""
    declared_failed = declared_failed or {}
    assert set(declared_failed) <= set(arms), \
        f'--failed names {sorted(set(declared_failed) - set(arms))}, which are not among the scored {list(arms)}'
    ctx = ctx or E.load_context()
    p_sha, labels = ctx['p_sha'], ctx['labels']
    base = E.base_images(p_sha)
    runs = {v: read_run(outputs, v, p_sha, declared_failed, smoke) for v in arms}
    ok = [v for v in arms if runs[v]['status'] == 'ok']
    label_sha = {a: rb_pick.array_sha256(labels[a]) for a in ASPECTS}
    img_sha = {BASE: {a: rb_pick.array_sha256(base[a]) for a in ASPECTS}}
    img_sha.update({v: {a: rb_pick.array_sha256(runs[v]['img'][a]) for a in ASPECTS} for v in ok})
    identity = dict(p_sha256=p_sha, label_sha256=label_sha, n_draws=n_draws, arms=list(arms), img_sha256=img_sha,
                    run_sha256={v: runs[v]['run_sha256'] for v in arms},
                    failed={v: runs[v]['cause'] for v in arms if runs[v]['status'] == 'failed'})
    effects = {v: {} for v in ok}
    base_A = {}
    for a in ASPECTS:
        imgs = {BASE: base[a], **{v: runs[v]['img'][a] for v in ok}}
        parts = {n: fixed_partitions(outputs, n, a, imgs[n], p_sha) for n in imgs}
        if n_draws is None:
            labelled = np.flatnonzero(labels[a] >= 0)
            point = {n: float(rb_ami.score(labels[a], parts[n], labelled)) for n in imgs}
            eff = {n: dict(A=point[n], e=point[n] - point[BASE]) for n in imgs}
        else:
            cache = DiskBoots(outputs, a, n_draws, p_sha, label_sha[a], {n: img_sha[n][a] for n in imgs})
            eff = aspect_effects(labels[a], parts, BASE, n_draws, workers, cache)
        base_A[a] = eff[BASE]['A']
        for v in ok:
            effects[v][a] = eff[v]
        print(f'{a}: scored {len(imgs)} arms', flush=True)
    results = {}
    for v in arms:
        if runs[v]['status'] == 'failed':
            rec = dict(failed_record(runs[v]['cause']), **({'vg3_problem': runs[v]['vg3_problem']}
                                                           if 'vg3_problem' in runs[v] else {}))
        else:
            rec = condition_record(effects[v])
        results[v] = dict(rec, best_epoch=runs[v]['best_epoch'], **({'rerun_of': runs[v]['rerun_of']} if 'rerun_of' in runs[v] else {}))
    return dict(base=base_A, results=results, identity=identity)


def _header(ctx, n_draws, smoke, code):
    return dict(smoke=bool(smoke), n_draws=n_draws, reference=BASE, seed=SEED, t_g=V.T_G, promising=PROMISING,
                ranges=dict(RANGES), rule_commit=rg_paths.RULE_COMMIT, versions=rb_ami.versions(),
                label_counts={a: int((ctx['labels'][a] >= 0).sum()) for a in ASPECTS}, **code)


def _clean(record):
    return json.loads(json.dumps(record, allow_nan=False))


def check_score_code(path, code_sha256):
    """Refuse an existing output made by other code than `code_sha256` (the code the run started with; copied from
    `eg_score3.check_score_code`). The old file is only read."""
    old = json.loads(Path(path).read_text())
    assert old.get('code_sha256') == code_sha256, f'{path}: stale code_sha256; move {path} aside, then rerun'
    return old


def _write_or_skip(path, record):
    """Write `record`; an existing file with the same identity is kept (never overwritten), another one stops."""
    path = Path(path)
    if path.exists():
        old = check_score_code(path, record['code_sha256'])
        assert old['identity'] == record['identity'], f'{path} exists with other inputs; move it aside'
        print(f'{path}: exists with the same inputs, kept', flush=True)
        return old
    _json_dump(path, record)
    return record


def keep_check(path, record):
    """Write the check `record`, or keep an existing file that is the same check: same `code_sha256`, same inputs
    (`identity`, reference and P hashes) and same values. Any other existing file stops the run with a message to move it
    aside and rerun; the file is never overwritten or deleted here (VA1 and VB1 files gate later stages)."""
    path = Path(path)
    if path.exists():
        old = check_score_code(path, record['code_sha256'])
        assert all(old.get(k) == record.get(k) for k in ('identity', 'reference_sha256', 'p_sha256')), \
            f'{path} exists with other inputs; move it aside, then rerun'
        assert old['pass'] == record['pass'] and old['smoke'] == record['smoke'] and \
            all(abs(old['items'][k]['value'] - v['value']) < 1e-9 for k, v in record['items'].items()), \
            f'{path} exists with other values; move it aside, then rerun'
        print(f'{path}: exists with the same code, inputs and values, kept', flush=True)
        return old
    _json_dump(path, record)
    return record


# ------------------------------------------------------------------ modes
def ladder_records(outputs, n_draws, workers, declared_failed, ctx, smoke):
    """({rung: record} of every ladder rung with a run folder, in ladder order; the scored identity)."""
    rungs = [r for r in V.LADDER if r == 'GR-8' or (arm_folder(outputs, r) / 'run.json').exists()]
    scored = score_arms(outputs, rungs, n_draws, workers, declared_failed, ctx, smoke)
    return {r: scored['results'][r] for r in rungs}, scored


def run_climb(outputs, workers, smoke=False, declared_failed=None):
    """The climb's decision on the rungs run so far, appended to stage_a/climb.json (point estimates only)."""
    read_va1(outputs, smoke)
    code = code_state()                                   # taken before the scoring: the record names the code that computed it
    path = stage_dir(outputs) / 'climb.json'
    ctx = E.load_context()
    records, scored = ladder_records(outputs, None, workers, declared_failed, ctx, smoke)
    entry = dict(climb_entry(records), identity=scored['identity'], **code)
    old = json.loads(path.read_text()) if path.exists() else dict(stage='climb', smoke=bool(smoke), decisions=[])
    assert old['stage'] == 'climb' and old['smoke'] == bool(smoke), f'{path}: from a {"smoke" if old["smoke"] else "real"} run'
    log = extend_climb_log(old['decisions'], _clean(entry))
    if log != old['decisions']:
        _json_dump(path, _clean(dict(stage='climb', smoke=bool(smoke), t_g=V.T_G, promising=PROMISING, ranges=dict(RANGES),
                                     p_sha256=ctx['p_sha'], decisions=log)))
    return log[-1]


def read_training_va1(outputs, smoke):
    path = Path(outputs) / 'regression' / 'va1.json'
    assert path.exists(), f'VA1: {path} missing; run the training half first'
    va1 = json.loads(path.read_text())
    assert va1.get('smoke') is bool(smoke), f'VA1: {path}: smoke flag differs'
    assert smoke or va1.get('pass') is True, f'VA1 training half did not pass ({path})'
    return va1


def read_va1(outputs, smoke):
    """Both VA1 halves, bound to this scorer; the two-epoch training smoke cannot pass."""
    path = Path(outputs) / 'regression' / 'va1_scorer.json'
    assert path.exists(), f'VA1: {path} missing; run `check va1` first'
    scorer = json.loads(path.read_text())
    assert scorer.get('stage') == 'va1_scorer', f'VA1: {path}: stage differs'
    assert scorer.get('smoke') is bool(smoke), f'VA1: {path}: smoke flag differs'
    assert scorer.get('pass') is True, f'VA1 scorer half did not pass ({path})'
    assert scorer.get('code_sha256') == scorer_code_sha256(), f'VA1: {path}: stale code_sha256; rerun `check va1`'
    return read_training_va1(outputs, smoke)


def read_eq(outputs, smoke=False):
    """regression/eq.json's verdict: dict(pass (bool), sha256 of the file). GQ-8 trains only after EQ ran, so the file must
    exist, and it was made in the same mode as this scoring."""
    path = Path(outputs) / 'regression' / 'eq.json'
    assert path.exists(), f'{path} missing: EQ (rule §2.1) must have run before GQ-8'
    eq = json.loads(path.read_text())
    from rg_train import eq_code_sha256
    assert eq.get('eq_code_sha256') == eq_code_sha256(), f'EQ: {path}: stale or missing code hashes; move it aside and rerun EQ'
    assert isinstance(eq.get('pass'), bool), f'{path}: no boolean `pass`'
    assert eq.get('smoke') == bool(smoke), f'{path}: smoke {eq.get("smoke")!r}, this scoring has smoke={bool(smoke)}'
    return dict(**{'pass': eq['pass']}, sha256=_file_sha256(path))


def run_a(outputs, n_draws, workers, smoke=False, declared_failed=None):
    """Stage A's verdict -> stage_a/verdict.json: every arm of the finished climb plus A_OTHER, v*_A, the outcome."""
    read_va1(outputs, smoke)
    code = code_state()                                   # taken before the scoring: the record names the code that computed it
    climb_path = stage_dir(outputs) / 'climb.json'
    assert climb_path.exists(), f'{climb_path} missing: run `climb` first'
    climb = json.loads(climb_path.read_text())
    assert climb['smoke'] == bool(smoke), f'{climb_path}: smoke flag differs'
    rungs = climb_rungs(climb['decisions'])
    arms = rungs + list(V.A_OTHER)
    check_arm_dirs(outputs, [a for a in arms if a != 'GR-8'])
    eq = read_eq(outputs, smoke)
    ctx = E.load_context()
    scored = score_arms(outputs, arms, n_draws, workers, declared_failed, ctx, smoke)
    results = scored['results']
    for r, x in climb['decisions'][-1]['rungs'].items():            # the scorer reproduces the climb's own numbers
        y = results[r]
        assert (x['status'], x['guarded']) == (y['status'], y['guarded']) and \
            (x['A_genre'] is None or abs(x['A_genre'] - y['A_genre']) < 1e-9), f'{r}: differs from climb.json'
    d = decide_a(results, eq['pass'], rungs)
    record = dict(stage='a', **_header(ctx, n_draws, smoke, code), base_A=scored['base'], arms=list(results), climb_rungs=rungs,
                  climb_reason=climb['decisions'][-1]['reason'], still_rising_at_cap=climb['decisions'][-1]['still_rising'],
                  results={n: results[n] for n in V.A_TIE_ORDER if n in results}, **d, stage_b_new=list(V.B_NEW),
                  identity=dict(scored['identity'], eq=eq))      # a verdict made with another eq.json is not kept
    return _write_or_skip(stage_dir(outputs) / 'verdict.json', _clean(record))


def run_check_va1(outputs, n_draws, workers, smoke=False):
    """VA1 scorer half: the scorer gives 12-25's score_all.json `results` of R6-5 (`base_A`) and GR-8 (each A, e and bound)
    and T_g within 1e-6 -> regression/va1_scorer.json. A smoke (20 draws) checks A and e only, not the bounds."""
    import rb_part1_report
    code = code_state()                                   # taken before the scoring: the record names the code that computed it
    path = Path(outputs) / 'regression' / 'va1_scorer.json'
    ref_path = rg_paths.EG_III / 'score_all.json'
    ref = json.loads(ref_path.read_text())
    assert ref['stage'] == 'all' and not ref['smoke'] and ref['n_draws'] == rb_ami.N_DRAWS and ref['seed'] == SEED
    assert ref['t_g'] == V.T_G, 'rg_variants.T_G differs from 12-25\'s score_all.json'
    ctx = E.load_context()
    scored = score_arms(outputs, ['GR-8'], n_draws, workers, None, ctx, smoke)
    assert scored['results']['GR-8']['status'] == 'ok'
    items = {}

    def item(name, value, want):
        items[name] = dict(value=float(value), expected=want, diff=float(abs(value - want)), **{'pass': bool(abs(value - want) <= TOL)})

    for a in ASPECTS:
        item(f'{BASE}_{a}_A', scored['base'][a], ref['base_A'][a])
        x, y = scored['results']['GR-8']['effects'][a], ref['results']['GR-8']['effects'][a]
        for k in ('A', 'e') + (() if smoke else ('lo', 'hi')):
            item(f'GR-8_{a}_{k}', x[k], y[k])
    assert V.T_G == json.loads((rg_paths.RB_PART1 / 'reported.json').read_text())['references_p']['genre_word']['genre']
    item('T_g', rb_part1_report.kmeans_ami(rb_data.p_instructed_images('genre_word'), ctx['labels']['genre']), V.T_G)
    for k, v in items.items():
        print(f'{k}: {v["value"]:.10f} (12-25 {v["expected"]:.10f}, diff {v["diff"]:.1e})', flush=True)
    record = dict(stage='va1_scorer', smoke=bool(smoke), n_draws=n_draws, bounds_checked=not smoke,
                  **{'pass': all(i['pass'] for i in items.values())}, tol=TOL, items=items, reference=str(ref_path),
                  reference_sha256=_file_sha256(ref_path), p_sha256=ctx['p_sha'], versions=rb_ami.versions(), **code,
                  identity=scored['identity'])
    return keep_check(path, _clean(record))


# ------------------------------------------------------------------ stage B: files
def b_dir(outputs):
    return Path(outputs) / 'stage_b'


def read_vb1(outputs, smoke):
    """The gate on `b`: stage_b/vb1.json exists, passed, is of the same kind (smoke or real) and was made by this scorer
    code (a later edit of the scorer needs `check vb1` again)."""
    path = b_dir(outputs) / 'vb1.json'
    assert path.exists(), f'{path} missing: run `check vb1` first (VB1)'
    vb1 = json.loads(path.read_text())
    assert vb1['stage'] == 'vb1' and vb1['smoke'] == bool(smoke), f'{path}: stage or smoke flag differs'
    assert vb1['pass'] is True, f'VB1 did not pass ({path}); stages B and C stop, trace the cause'
    assert vb1['code_sha256'] == scorer_code_sha256(), f'{path}: stale code_sha256; rerun `check vb1`'
    return vb1


def fresh_cache_dir(outputs, arm, seed):
    return b_dir(outputs) / 'partitions' / arm / f's{seed}'


def fresh_partitions(outputs, arm, aspect, img, p_sha, seed):
    """int64 [3, n]: the arm's fixed partitions of one aspect on a fresh seed, cached under stage_b/partitions/ (copied
    from `eg_score3.fixed_partitions`, whose cache sits under stage_iii/)."""
    path = fresh_cache_dir(outputs, arm, seed) / f'{aspect}__partitions.npz'
    identity = E._identity(arm, aspect, p_sha, rb_pick.array_sha256(img), seed)
    parts = E._read_npz_with_record(path, 'labels', identity)
    if parts is None:
        parts = rb_ami.partitions(img)
        E._write_npz_with_record(path, 'labels', parts, dict(identity=identity, versions=rb_ami.versions(),
                                                             k=rb_ami.K, states=list(rb_ami.STATES)))
    assert parts.shape == (3, len(img)) and parts.dtype == np.int64
    return parts


class FreshBoots:
    """Per-arm bootstrap scores of one aspect on one fresh seed, cached under stage_b/partitions/ (copied from
    `eg_score3.DiskBoots`, whose cache sits under stage_iii/). The identity says the draws are the fresh seeds' (I_b
    with S_b between them), not seed 0's."""

    def __init__(self, outputs, aspect, n_draws, p_sha, label_sha, img_sha, seed):
        self.outputs, self.aspect, self.n_draws, self.seed = outputs, aspect, n_draws, seed
        self.p_sha, self.label_sha, self.img_sha = p_sha, label_sha, img_sha

    def _where(self, name):
        path = fresh_cache_dir(self.outputs, name, self.seed) / f'{self.aspect}__boot.npz'
        return path, E._identity(name, self.aspect, self.p_sha, self.img_sha[name], self.seed, label_sha256=self.label_sha,
                                 n_draws=self.n_draws, draws=E.FRESH_TAG)

    def get(self, name):
        path, identity = self._where(name)
        return E._read_npz_with_record(path, 'boot', identity)

    def put(self, name, boot):
        path, identity = self._where(name)
        E._write_npz_with_record(path, 'boot', np.asarray(boot, dtype=np.float64), dict(identity=identity))


def read_fresh_run(outputs, arm, seed, p_sha, declared_failed, smoke):
    """{status, cause, best_epoch, run_sha256, img {aspect: array}} of one arm on a fresh seed (copied from `read_run`, which
    is seed 0's, and `eg_score3.read_fresh_run`). GR-8 and R6-5 are 12-25's frozen runs, every other arm is this round's
    (`run_dir`). Failed run: no exports read. An arm in `declared_failed` has its exports read and must fail VB3 on at
    least one aspect of this seed (VB3 is the per-export VA3)."""
    assert seed in FRESH, f'VB4: seed {seed} is not one of {tuple(FRESH)}'
    folder = rg_paths.frozen_run(arm, seed) if arm in FROZEN_ARMS else rg_paths.run_dir(outputs, arm, seed)
    run_path = folder / 'run.json'
    assert run_path.exists(), f'{arm} s{seed}: {run_path} missing (the run has not finished)'
    run = json.loads(run_path.read_text())
    if arm not in FROZEN_ARMS:
        check_run_code(outputs, run, smoke)
    assert run['variant'] == arm and run['seed'] == seed, (arm, seed, run.get('variant'), run.get('seed'))
    want = V.eg_form(V.ARMS[arm]) if arm in FROZEN_ARMS else V.switches(arm)    # as ticket 03's `read_run`
    assert run['switches'] == want, f'{arm} s{seed}: run.json switches {run["switches"]}, expected {want}'
    assert run['status'] in ('ok', 'failed'), (arm, seed, run['status'])
    assert p_sha == rg_paths.P_SHA256, 'P differs from 12-25\'s'
    assert run.get('p_sha256') == p_sha, f'{arm} s{seed}: run.json p_sha256 {run.get("p_sha256")!r} is not 12-25\'s P'
    assert smoke or not run.get('smoke'), f'{arm} s{seed}: a smoke run is not scored outside a smoke'
    rec = dict(status=run['status'], cause=run.get('cause'), best_epoch=run.get('best_epoch'),
               run_sha256=_file_sha256(run_path), img=None, **({'rerun_of': run['rerun_of']} if 'rerun_of' in run else {}))
    if run['status'] == 'failed':
        assert run.get('cause'), f'{arm} s{seed}: failed run without a cause'
        # as ticket 03's `read_run`: rule §4.2, a run fails if its loss goes non-finite, or if it still fails VB3 after its
        # (one) rerun; this round's trainer spells the second kind 'va3'
        assert run.get('failure_kind') in ('nonfinite_loss', 'va3'), (arm, seed, 'failure_kind', run.get('failure_kind'))
        assert run['failure_kind'] != 'va3' or 'rerun_of' in run, \
            f'{arm} s{seed}: failed VB3 and has not been rerun; rerun it once first'
        return rec
    img, problems = {}, {}
    for a in ASPECTS:
        with np.load(folder / f'{a}__P.npz', allow_pickle=False) as z:
            assert 'img' in z.files, (arm, seed, a, z.files)
            img[a] = z['img']
        problem = rb_pick.vp2_problem(img[a])                     # VB3: 29,232 finite unit-norm rows
        if problem is not None:
            problems[a] = problem
    if arm in declared_failed:
        assert problems, f'VB3 {arm} s{seed}: --failed declared it, but its exports pass VB3 on every aspect (rule §4.2)'
        return dict(rec, status='failed', cause=declared_failed[arm], vg3_problem=problems)
    assert not problems, f'VB3 {arm} s{seed}: {problems} (rerun it, or declare the failure with --failed)'
    return dict(rec, img=img)


def read_arm_runs(outputs, arm, p_sha, declared_failed, smoke):
    """{seed: record} of one arm on seeds 1 to 3. An arm in `declared_failed` is declared failed only on the seeds whose
    exports fail VB3 (its other seeds are read as they are; at least one seed must fail)."""
    runs = {}
    for s in FRESH:
        try:
            runs[s] = read_fresh_run(outputs, arm, s, p_sha, {}, smoke)
        except AssertionError as err:
            if arm in declared_failed and str(err).startswith('VB3'):
                runs[s] = read_fresh_run(outputs, arm, s, p_sha, declared_failed, smoke)
            else:
                raise
    if arm in declared_failed:
        assert any(r['status'] == 'failed' for r in runs.values()), f'--failed {arm}: no seed of {arm} is a failed run'
    return runs


def score_fresh_arms(outputs, arms, n_draws, workers, declared_failed, ctx, smoke):
    """{'records': {arm: b_record | failed_b_record}, 'identity': {...}} for `arms` against R6-5 (seeds 1 to 3). R6-5's runs
    must all be ok. An arm with a failed run is not scored (rule §4.2 "Failed runs")."""
    declared_failed = declared_failed or {}
    p_sha, labels = ctx['p_sha'], ctx['labels']
    assert BASE not in arms and set(declared_failed) <= set(arms), (arms, sorted(declared_failed))
    runs = {n: read_arm_runs(outputs, n, p_sha, declared_failed, smoke) for n in list(arms) + [BASE]}
    base_failed = arm_run_status(runs[BASE])
    assert base_failed[0] == 'ok', f'{BASE} (the comparator) has a failed fresh-seed run: {base_failed[1]}'
    status = {n: arm_run_status(runs[n]) for n in arms}
    ok = [n for n in arms if status[n][0] == 'ok']
    label_sha = {a: rb_pick.array_sha256(labels[a]) for a in ASPECTS}
    img_sha = {n: {str(s): {a: rb_pick.array_sha256(runs[n][s]['img'][a]) for a in ASPECTS} for s in FRESH} for n in [BASE] + ok}
    identity = dict(p_sha256=p_sha, label_sha256=label_sha, n_draws=n_draws, arms=list(arms), seeds=list(FRESH),
                    img_sha256=img_sha, run_sha256={n: {str(s): r['run_sha256'] for s, r in runs[n].items()} for n in runs},
                    failed={n: status[n][1] for n in arms if status[n][0] == 'failed'})
    effects = {n: {} for n in ok}
    for a in ASPECTS:
        parts = {n: {s: fresh_partitions(outputs, n, a, runs[n][s]['img'][a], p_sha, s) for s in FRESH} for n in [BASE] + ok}
        for n in ok:
            caches = {s: FreshBoots(outputs, a, n_draws, p_sha, label_sha[a], {m: img_sha[m][str(s)][a] for m in (n, BASE)}, s)
                      for s in FRESH}
            effects[n][a] = fresh_effect(labels[a], {n: parts[n], BASE: parts[BASE]}, n, BASE, n_draws, workers, caches)
        print(f'{a}: fresh seeds scored for {ok}', flush=True)
    records = {n: b_record(effects[n], n) if n in ok else failed_b_record(status[n][1]) for n in arms}
    return dict(records=records, identity=identity, runs={n: {str(s): dict(status=r['status'], cause=r['cause'],
                                                                           best_epoch=r['best_epoch'],
                                                                           run_sha256=r['run_sha256'],
                                                                           **({'rerun_of': r['rerun_of']}
                                                                              if 'rerun_of' in r else {}))
                                                              for s, r in runs[n].items()} for n in runs})


def _header_b(ctx, n_draws, smoke, code):
    return dict(smoke=bool(smoke), n_draws=n_draws, reference=BASE, seeds=list(FRESH), t_g=V.T_G, promising=PROMISING,
                ranges=dict(RANGES), rule_commit=rg_paths.RULE_COMMIT, versions=rb_ami.versions(),
                label_counts={a: int((ctx['labels'][a] >= 0).sum()) for a in ASPECTS}, **code)


def run_check_vb1(outputs, n_draws, workers, smoke=False):
    """VB1 (rule §4.2): the frozen fresh-seed exports of GR-8 and R6-5 hash to 12-25's `score_fresh.json`
    `identity.img_sha256` per arm, seed and aspect, and this scorer on them gives its `effects` (Ā, ē, both bounds per
    aspect) within 1e-6 -> stage_b/vb1.json. A smoke (20 draws) checks hashes, Ā and ē only, not the bounds."""
    code = code_state()                                   # taken before the scoring: the record names the code that computed it
    path = b_dir(outputs) / 'vb1.json'
    ref_path = rg_paths.EG_III / 'score_fresh.json'
    ref = json.loads(ref_path.read_text())
    ctx = E.load_context()
    assert ref['stage'] == 'fresh' and not ref['smoke'] and ref['n_draws'] == rb_ami.N_DRAWS and ref['winner'] == 'GR-8'
    assert ref['seeds'] == list(FRESH) and ref['effects'] and ref['identity']['p_sha256'] == ctx['p_sha']
    assert ref['identity']['label_sha256'] == {a: rb_pick.array_sha256(ctx['labels'][a]) for a in ASPECTS}
    scored = score_fresh_arms(outputs, ['GR-8'], n_draws, workers, None, ctx, smoke)
    rec = scored['records']['GR-8']
    assert rec['status'] == 'ok'
    check_frozen_hashes(scored['identity']['img_sha256'], ref['identity']['img_sha256'])
    items = compare_effects(rec['effects'], ref['effects'], ASPECTS, smoke)
    for k, v in items.items():
        print(f'{k}: {v["value"]:.10f} (12-25 {v["expected"]:.10f}, diff {v["diff"]:.1e})', flush=True)
    record = dict(stage='vb1', smoke=bool(smoke), n_draws=n_draws, bounds_checked=not smoke, hashes_checked=True,
                  **{'pass': all(i['pass'] for i in items.values())}, tol=TOL, items=items, reference=str(ref_path),
                  reference_sha256=_file_sha256(ref_path), p_sha256=ctx['p_sha'], versions=rb_ami.versions(), **code,
                  identity=scored['identity'])
    return keep_check(path, _clean(record))


def run_b(outputs, n_draws, workers, smoke=False, declared_failed=None):
    """Stage B's verdict -> stage_b/verdict.json: GR-8, GR-2, GR-4 and v*_A (A-NEW) on seeds 1 to 3 against R6-5."""
    code = code_state()                                   # taken before the scoring: the record names the code that computed it
    declared_failed = declared_failed or {}
    vb1 = read_vb1(outputs, smoke)
    va = read_stage_a(outputs, smoke)
    arms = b_arms(va)
    v_star = va['winner'] if va['outcome'] == 'A-NEW' else None
    check_vb4(arms, v_star, FRESH)
    new = [a for a in arms if a not in FROZEN_ARMS]
    check_fresh_dirs(outputs, new)
    assert set(declared_failed) <= set(new), f'VB4: --failed {sorted(declared_failed)}: only {new} run on fresh seeds'
    ctx = E.load_context()
    assert va['identity']['p_sha256'] == ctx['p_sha'], 'stage A\'s verdict is on another P'
    scored = score_fresh_arms(outputs, list(arms), n_draws, workers, declared_failed, ctx, smoke)
    ref = json.loads((rg_paths.EG_III / 'score_fresh.json').read_text())
    check_frozen_hashes({n: scored['identity']['img_sha256'][n] for n in FROZEN_ARMS if n in scored['identity']['img_sha256']},
                        ref['identity']['img_sha256'])                  # VB1 still holds for the files just read
    d = decide_b(scored['records'], v_star)
    identity = dict(scored['identity'], vb1_sha256=_file_sha256(b_dir(outputs) / 'vb1.json'),
                    stage_a_verdict_sha256=_file_sha256(stage_dir(outputs) / 'verdict.json'))
    record = dict(stage='b', **_header_b(ctx, n_draws, smoke, code), a_outcome=va['outcome'], v_star_a=va['winner'],
                  stage_b_arms=list(arms), vb1_pass=vb1['pass'], results=scored['records'], runs=scored['runs'], **d,
                  identity=identity)
    return _write_or_skip(b_dir(outputs) / 'verdict.json', _clean(record))


def read_stage_a(outputs, smoke):
    """stage_a/verdict.json (ticket `a`'s): exists, a stage A record of the same kind."""
    path = stage_dir(outputs) / 'verdict.json'
    assert path.exists(), f'{path} missing: stage A\'s verdict must exist before stage B'
    verdict = json.loads(path.read_text())
    assert verdict['stage'] == 'a' and verdict['smoke'] == bool(smoke), f'{path}: stage or smoke flag differs'
    return verdict


# ------------------------------------------------------------------ CLI
def parse_failed(items):
    """['ARM=CAUSE', ...] -> {arm: cause}: an arm still failing VA3 after its rerun."""
    out = {}
    for item in items:
        arm, sep, cause = item.partition('=')
        assert sep and cause, f'--failed wants ARM=CAUSE, got {item!r}'
        assert arm in V.ARMS and arm != 'GR-8', f'--failed {arm}: not a new arm of this stage'
        assert arm not in out, f'--failed twice for {arm}'
        out[arm] = cause
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Stage A (seed 0) and stage B (seeds 1 to 3) scorer.')
    parser.add_argument('mode', choices=['climb', 'a', 'b', 'check', 'report'])
    parser.add_argument('what', nargs='?', choices=['va1', 'vb1'], help='for `check`')
    parser.add_argument('--outputs', type=Path, default=rg_paths.OUTPUTS)
    parser.add_argument('--smoke', action='store_true', help='20 bootstrap draws; outputs under /tmp/claude-0/')
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    parser.add_argument('--failed', action='append', default=[], metavar='ARM=CAUSE',
                        help='an arm still failing VA3 after its forced rerun (repeatable)')
    args = parser.parse_args(argv)
    assert (args.mode == 'check') == (args.what is not None), 'usage: climb | a | b | report | check va1 | check vb1'
    if args.smoke:
        assert rg_paths.smoke_ok(args.outputs), 'smoke needs --outputs under /tmp/claude-0/'
    rg_paths.check_imports()
    n_draws = 20 if args.smoke else rb_ami.N_DRAWS
    failed = parse_failed(args.failed)
    assert not (args.mode == 'check' and failed), '--failed does not apply to `check`'
    if args.mode == 'report':
        assert not failed, '--failed does not apply to `report`'
        import rg_report                                 # lazy: the report code is not part of the scorer's code hash
        rec = rg_report.run_report(args.outputs, args.smoke)
        print(f"{args.outputs}/reported_ab.md: stage A {rec['a_outcome']}, v*_A {rec['v_star_a']}; stage B "
              f"{rec['stage_b']['outcome'] if rec['stage_b'] else 'not yet'}")
    elif args.mode == 'check' and args.what == 'vb1':
        rec = run_check_vb1(args.outputs, n_draws, args.workers, args.smoke)
        print(f"{b_dir(args.outputs) / 'vb1.json'}: VB1 {'PASS' if rec['pass'] else 'FAIL'}; "
              f"{sum(i['pass'] for i in rec['items'].values())} of {len(rec['items'])} items within {TOL}")
    elif args.mode == 'check':
        rec = run_check_va1(args.outputs, n_draws, args.workers, args.smoke)
        print(f"{args.outputs}/regression/va1_scorer.json: VA1 scorer half {'PASS' if rec['pass'] else 'FAIL'}; "
              f"{sum(i['pass'] for i in rec['items'].values())} of {len(rec['items'])} items within {TOL}")
    elif args.mode == 'b':
        rec = run_b(args.outputs, n_draws, args.workers, args.smoke, failed)
        print(f"{b_dir(args.outputs) / 'verdict.json'}: {rec['outcome']}, v_held {rec['v_held']}; confirmed {rec['confirmed']}")
    elif args.mode == 'climb':
        e = run_climb(args.outputs, args.workers, args.smoke, failed)
        print(f"{stage_dir(args.outputs) / 'climb.json'}: newest {e['newest']} ({e['reason']}); next rung "
              f"{e['next'] or 'none'}")
    else:
        rec = run_a(args.outputs, n_draws, args.workers, args.smoke, failed)
        print(f"{stage_dir(args.outputs) / 'verdict.json'}: v*_A {rec['winner']}, {rec['outcome']}, "
              f"A_genre {rec['A_genre_winner']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
