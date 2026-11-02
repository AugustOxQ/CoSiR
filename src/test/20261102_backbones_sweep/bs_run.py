"""r14 stage runner (ticket 12a): spec §3.5 stages 0 to 5, §3.7, D11, D18, D21. Plumbing only: every number comes from the
round's modules, called through their command lines (one subprocess per step, so a crash is a failed step, not a dead
runner). Each step logs to OUT/stage<k>/logs/<step>.log, appends {step, argv, rc, seconds} to OUT/stage<k>/steps.jsonl and
prints one line `BS_RUN <stage> <step> rc=<rc> seconds=<s>`.

    python bs_run.py stage0 [--parts cpu|gpu|all] [--only ID ...] [--resolve ID=ok|failed ...] [--close-pending]
        the gates (i) to (xii) (ids below); evidence in OUT/stage0/gate.json (merged across runs, so the CPU and the GPU
        parts can run apart); writes RECORDS/status.json once every gate and the part a runs are in (see `status`).
        Gates (iv), (v), (ix) and the extraction parity (i) need the GPU: run them under `flock -o -w <s> /tmp/gpu0.lock`.
        D18 branches: a failed vii-sha (an r10 ablation export) or vii-matched (a matched arm) is 'pending', not failed,
        until it is resolved: CPU re-export from the weights (§3.2) or the node rerun (after part a), then
        `stage0 --resolve vii-sha=ok` (or `=failed`). `--close-pending` turns every pending gate into failed.
    python bs_run.py status [--close-pending]   assemble RECORDS/status.json (schema: bs_outcome.py STATUS SCHEMA)
    python bs_run.py exports --set qwen         EmoSet exports (val, test; seeds 0 to 4) of photo_qwen (r13's five photo runs) and
        cstar_qwen (the frozen c*), through bs_photo export. Before stage 2.
    python bs_run.py exports --set part-a --enc E [E ...]    after the node's part a is pulled: the X-c* held export of E (bs_cstar
        export --kind held --device cpu) and the photo_E exports on Affection and EmoSet (val, test), seeds 0 to 4. Skips the
        arms that part_a_state(E) marks failed (and says so). Before stage 1 (Affection val) and stage 3 (held, Affection test)
        and stage 2 (EmoSet). `--seeds` (default 0 to 4) is for a dry check. All exports run on the local CPU (every store is
        local; gate 0(ix)'s c* part passed on CPU, max abs diff 6.1e-05). A job whose files exist with a matching sha256 is
        skipped; a stage that needs a missing export refuses and names this step.
    python bs_run.py stage1 [--enc E ...]       per encoder still in part a: seed ranges and read-out picks (tickets 05 to 07)
        -> RECORDS/stage1_<enc>.json, committed before any held number of that encoder. Needs the part-a exports.
    python bs_run.py stage2                     r_E, read-out and wording picks -> RECORDS/stage2_emoset.json, committed. Runs AFTER
        part a and the exports: refuses unless, for each encoder, OUT/node/part_a_<enc>.json exists and either its photo_<enc>
        EmoSet exports exist or its photo runs failed, and the Qwen EmoSet exports exist (so the X photo models' rows are scored).
    python bs_run.py stage3 [--enc E ...] [--n-clu N --n-ret N]
        status.json, then the cells of every encoder and path that is not failed (refuses without the committed stage 1
        record, and the record must equal the stage 1 files the scorers read; refuses without the exports), part c (refuses
        without the committed stage 2 record). Writes the CELLS ONLY (OUT/stage3/cells_*.json) and prints the sha256 of each
        cells file and of the committed stage 1 and 2 records (also OUT/stage3/c13_sha256.json). No verdict.
    python bs_run.py decide [--smoke]           after C13 (§3.4: Codex re-derives "before any verdict is written"). Refuses unless
        RECORDS/c13.json is committed in git HEAD with "pass": true and "sha256": {"stage3/<cells file>": hex, ...,
        "records/<stage 1 or 2 record>": hex, ...} equal to the current files. Then `bs_outcome decide` -> OUT/stage3/outcomes.json
        and RECORDS/verdicts.json (committed). The re-derivation itself is the run chat's Codex job, not code.
    Order: stage0 (cpu, gpu) -> node (viii, prep, part a, sweeps) -> exports qwen, exports part-a -> stage1 -> stage2 -> stage3
    -> C13 -> decide -> stage4 -> stage5.
    python bs_run.py stage4                     the three part b sheets (bs_partb: knob, photo, ablation).
    python bs_run.py diag [--smoke]             bs_diag.run_all(smoke=...) (ticket 12b; imported here, lazily).
    python bs_run.py stage5 [--smoke]           the same call (stage 5 is the diagnostics, §3.5/§3.6).
    python bs_run.py standin                    the Qwen stand-in scores (bs_artclu/bs_artret/bs_affection `--enc qwen`);
        refuses the production OUT: set BS_OUTPUTS to its own folder (its cells must never reach OUT/stage3).

Switches (smoke and tests; unset in the real run):
    BS_RUN_NO_COMMIT=1   no git add/commit, and the committed-record guards check the file only. Also `--no-commit`.
    BS_RECORDS=<dir>     records folder (default bs_paths.RECORDS); a smoke points it at the scratch folder.
    BS_RUN_STUB=1        steps are not executed (rc 0, or 1 for the step names in BS_RUN_FAIL, comma list): wiring check.
Commit rule (§3.7): records are committed in the checkout that holds this code (`bs_paths.REPO`), `git add -f` (`*.json`
is gitignored), scoped to the files named, with the attribution lines of the run chat's commit message in BS_RUN_TRAILER.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import bs_paths as BP

ROUND = BP.ROUND
OUT = Path(BP.OUT)
RECORDS = Path(os.environ.get('BS_RECORDS') or BP.RECORDS)
ENCS = tuple(BP.NEW_ENCODERS)
P_SEEDS = tuple(BP.SEEDS)
PATHS = ('paintings', 'photos')
NO_COMMIT = bool(os.environ.get('BS_RUN_NO_COMMIT'))
STUB = bool(os.environ.get('BS_RUN_STUB'))
ART_VIEW = {'siglip2': 'siglip2', 'siglip1': 'siglip1'}
MATCHED_ENC = {v: k for k, v in BP.MATCHED.items()}        # M2 -> siglip2
FROZEN_ENC = {v: k for k, v in BP.FROZEN_ARM.items()}      # F2 -> siglip2


# ============================================================================================== steps
def now():
    return time.strftime('%F %H:%M')


def py(module, *args):
    return [sys.executable, str(ROUND / f'{module}.py'), *map(str, args)]


def step(stage, name, argv, extra_env=None):
    """Run one command; log, time and record it. Returns {step, argv, rc, seconds, log}."""
    stage = f'stage{stage}' if isinstance(stage, int) else stage            # 'exports' has its own folder
    d = OUT / stage
    (d / 'logs').mkdir(parents=True, exist_ok=True)
    log = d / 'logs' / f'{name}.log'
    t0 = time.time()
    if STUB:
        rc = 1 if name in os.environ.get('BS_RUN_FAIL', '').split(',') else 0
        log.write_text(f'stub: {argv}\n')
    else:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', **(extra_env or {}))
        with open(log, 'w') as fh:
            rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROUND)).returncode
    rec = dict(step=name, argv=[str(a) for a in argv], rc=rc, seconds=round(time.time() - t0, 1), log=str(log), at=now())
    with open(d / 'steps.jsonl', 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
    print(f'BS_RUN {stage} {name} rc={rc} seconds={rec["seconds"]}', flush=True)
    return rec


def jload(path):
    return json.loads(Path(path).read_text())


def jdump(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


# ============================================================================================== records and git
def git_show(f):
    r = subprocess.run(['git', '-C', str(BP.REPO), 'show', f'HEAD:{Path(f).resolve().relative_to(BP.REPO.resolve())}'],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def committed(f):
    """'' when the record file `f` exists and (unless NO_COMMIT) equals its content in git HEAD, else the reason."""
    f = Path(f)
    if not f.is_file():
        return f'{f} does not exist'
    if NO_COMMIT:
        return ''
    head = git_show(f)
    if head is None:
        return f'{f} is not in git HEAD: commit it first'
    return '' if head == f.read_bytes() else f'{f} differs from its content in git HEAD: commit it first'


def require(f, what):
    why = committed(f)
    if why:
        raise SystemExit(f'REFUSED: {what}: {why}')


def commit(files, message):
    """`git add -f` and a scoped commit of exactly `files` (no other path), in the code's checkout. No-op under NO_COMMIT or
    when the files already equal HEAD."""
    files = [Path(f) for f in files]
    if NO_COMMIT:
        print(f'BS_RUN commit skipped (BS_RUN_NO_COMMIT): {[f.name for f in files]}', flush=True)
        return
    todo = [f for f in files if committed(f)]
    if not todo:
        return
    rel = [str(f.resolve().relative_to(BP.REPO.resolve())) for f in todo]
    trailer = os.environ.get('BS_RUN_TRAILER', '')
    subprocess.run(['git', '-C', str(BP.REPO), 'add', '-f', '--', *rel], check=True)
    subprocess.run(['git', '-C', str(BP.REPO), 'commit', '-m', message + (f'\n\n{trailer}' if trailer else ''), '--', *rel],
                   check=True)
    print(f'BS_RUN committed {rel}', flush=True)


# ============================================================================================== stage 0
def gate_table():
    """(id, step name, argv, kind). `kind` cpu|gpu. Effects of a failure are in `effects`."""
    G = []
    for case in ('clip_l14_affection', 'qwen_default_affection', 'emotion_focus_affection', 'emotion_focus_photo_affection',
                 'siglip2_artelingo', 'siglip1_artelingo'):
        G.append((f'i:{case}', py('bs_extract', 'parity', '--cases', case), 'gpu'))
    G += [('ii', py('bs_extract', 'check'), 'cpu'),
          ('iii', py('bs_emoset', 'nearcopy', '--out', OUT / 'stage0/nearcopy_emoset.json'), 'cpu'),
          ('iv', py('bs_cstar', 'regress-iv'), 'gpu'),
          ('v', py('bs_photo', 'regress-v'), 'gpu'),
          ('vi:artclu', py('bs_artclu', 'replay', 'vi'), 'cpu'),
          ('vi:artret', py('bs_artret', 'replay', 'vi'), 'cpu'),
          ('vi:artret-ranges', py('bs_artret', 'replay', 'vi-ranges'), 'cpu'),
          ('vi:artret-pick', py('bs_artret', 'replay', 'vi-pick'), 'cpu'),
          ('vi:affection', py('bs_affection', 'replay', 'vi'), 'cpu'),
          ('vii-matched', py('bs_artclu', 'replay', 'vii-matched'), 'cpu'),
          ('vii-sha', py('bs_artclu', 'replay', 'vii-sha'), 'cpu'),
          ('vii-removals', py('bs_artclu', 'replay', 'vii-removals'), 'cpu'),
          ('ix:cstar', py('bs_cstar', 'check-export'), 'gpu'),
          ('ix:photo', py('bs_photo', 'check-export'), 'gpu'),
          ('x', py('bs_emoset', 'gate0x'), 'cpu'),
          ('xi', py('bs_emoset_score', 'gate0xi', '--recompute'), 'cpu'),     # the real check: --recompute (controller note)
          ('xii:f', py('bs_artclu', 'replay', 'xii-f', '--recompute', '--loader', 'bs_stores'), 'cpu'),
          ('xii:k', py('bs_artclu', 'replay', 'xii-k'), 'cpu'),
          ('xii:artret', py('bs_artret', 'replay', 'xii'), 'cpu'),
          ('xii:artret-pick', py('bs_artret', 'replay', 'xii-pick'), 'cpu')]
    return G


# Gates whose failure stops a path of every encoder, or part c. (viii) has no gate. (i) and (ii) are per store, below.
EFFECT = {'iii': ['part_c'], 'x': ['part_c'], 'xi': ['part_c'],
          'iv': ['paintings', 'photos'],                 # the trainer both paths use (X photo starts from X-c*)
          'v': ['photos'], 'vi:affection': ['photos', 'part_c'],      # part c: its exports are bs_photo.export_arm's, and it
          'ix:photo': ['photos', 'part_c'], 'ix:cstar': ['paintings']}  # reuses the Affection scoring code (final review F4)
for _g in ('vi:artclu', 'vi:artret', 'vi:artret-ranges', 'vi:artret-pick', 'vii-matched', 'vii-sha', 'vii-removals',
           'xii:f', 'xii:k', 'xii:artret', 'xii:artret-pick'):
    EFFECT[_g] = ['paintings']
BRANCH = {'vii-sha': 'cpu_reexport', 'vii-matched': 'matched_rerun_node'}     # D18 / §3.5 0(vii)


def parity_effect(case):
    """(i): the store under test decides: SigLIP rows of ArtELingo -> that encoder's paintings; CLIP L/14's Affection store ->
    its photos; the Qwen views (teacher and photo inputs) -> photos of all encoders."""
    if case.endswith('_artelingo'):
        return [f'{case.split("_")[0]}:paintings']
    if case.startswith('clip_l14'):
        return ['clip_l14:photos']
    return [f'{e}:photos' for e in ENCS]


def store_effects(res):
    """(ii) per store from OUT/stage0/stores.json (written by bs_extract check): the new view of an encoder -> its paths; the
    Qwen views and the EmoSet unpack -> photos of all / part c."""
    eff = set()
    if not res.get('emoset_unpack', {}).get('ok', True):
        eff.add('part_c')
    for s in res.get('stores', []):
        if s['ok']:
            continue
        ds, view = s['dataset'], s['view']
        if ds == 'emoset':
            eff.add('part_c')
        elif view in ENCS:
            eff.add(f'{view}:paintings' if ds == 'artelingo' else f'{view}:photos')
        else:
            eff.update(f'{e}:photos' for e in ENCS)
    return sorted(eff)


def expand(effects):
    """Effect names -> set of 'enc:path' and 'part_c'."""
    out = set()
    for e in effects:
        if e == 'part_c' or ':' in e:
            out.add(e)
        else:
            out.update(f'{enc}:{e}' for enc in ENCS)
    return out


def failed_items(gate_id):
    """Failing item names of a replay gate, from its replay file (empty when unreadable)."""
    f = OUT / 'replays' / f'replay_{gate_id.replace(":", "-")}.json'      # xii:f -> replay_xii-f.json (bs_artclu.py:619)
    try:
        items = jload(f)['items']
    except (OSError, KeyError, ValueError):
        return []
    return [k for k, v in items.items() if not v.get('ok', True)]


def gate_effects(gid, rec):
    """The set of failed 'enc:path' / 'part_c' names of one failed gate."""
    if gid.startswith('i:'):
        return expand(parity_effect(gid[2:]))
    if gid == 'ii':
        try:
            return expand(store_effects(jload(OUT / 'stage0/stores.json')))
        except (OSError, ValueError):
            return expand(['paintings', 'photos', 'part_c'])
    eff = EFFECT[gid]
    if gid == 'vii-matched':                                  # per encoder when the items say which matched arm failed
        encs = {MATCHED_ENC[k.split('/')[0]] for k in failed_items(gid) if k.split('/')[0] in MATCHED_ENC}
        if encs:
            return {f'{e}:paintings' for e in encs}
    if gid == 'xii:f':
        encs = {FROZEN_ENC[k.split('/')[0]] for k in failed_items(gid) if k.split('/')[0] in FROZEN_ENC}
        if encs:
            return {f'{e}:paintings' for e in encs}
    return expand(eff)


def gate_file():
    return OUT / 'stage0/gate.json'


def load_gates():
    return jload(gate_file()) if gate_file().exists() else {}


def run_stage0(a):
    gates = load_gates()
    for r in a.resolve or []:
        gid, _, val = r.partition('=')
        assert gid in gates and val in ('ok', 'failed'), f'--resolve {r}: gate must exist, value ok|failed'
        gates[gid]['resolved'] = val
        gates[gid]['resolved_at'] = now()
    t0 = time.time()
    table = gate_table()
    want = [g for g in table if (not a.only or g[0] in a.only or g[0].split(':')[0] in a.only)
            and (a.parts == 'all' or g[2] == a.parts)]
    assert not a.only or {x for x in a.only} <= {g[0] for g in table} | {g[0].split(':')[0] for g in table}, f'unknown gate in {a.only}'
    for gid, argv, kind in ([] if a.resolve and not a.only else want):
        r = step(0, gid.replace(':', '_'), argv)
        ok = r['rc'] == 0
        ent = dict(argv=r['argv'], kind=kind, rc=r['rc'], seconds=r['seconds'], ok=ok, log=r['log'], at=r['at'])
        if not ok:
            ent['failed_for'] = sorted(gate_effects(gid, r))
            if gid in BRANCH:
                ent['branch'] = BRANCH[gid]
                ent['failed_items'] = failed_items(gid)
        gates[gid] = ent
        jdump(gate_file(), gates)                                       # after each gate: a crash keeps the finished ones
    jdump(gate_file(), gates)
    viii = OUT / 'stage0/viii.json'
    print(f'BS_RUN stage0 gates={len(gates)} of {len(table)}; 0(viii) {"recorded" if viii.exists() else "not yet run (node)"}; '
          f'seconds={time.time() - t0:.1f}', flush=True)
    build_status(close_pending=a.close_pending, strict=False)
    failed = sorted(g for g, e in gates.items() if not e['ok'] and e.get('resolved') != 'ok')
    print(f'BS_RUN stage0 failed or pending gates: {failed or "none"}', flush=True)
    return 1 if failed else 0


# ============================================================================================== status (D18)
def gate_state(gates):
    """({'enc:path'|'part_c': [reasons]}, pending list, missing list). A gate that failed with a branch is pending until
    resolved; a resolution 'ok' clears it, 'failed' makes it a plain failure."""
    bad, pending = {}, []
    missing = [g[0] for g in gate_table() if g[0] not in gates]
    for gid, ent in gates.items():
        res = ent.get('resolved')
        if ent['ok'] or res == 'ok':
            continue
        if ent.get('branch') and res is None:
            pending.append(gid)
            continue
        for name in ent['failed_for']:
            bad.setdefault(name, []).append(gid)
    return bad, pending, missing


def part_a_state(enc):
    """(cstar_runs, photo_runs) from the node's OUT/node/part_a_<enc>.json: ok only with all five runs done. The two kinds
    are reported apart (a failed X photo run leaves cstar_runs ok, §6 / D18); failed X-c* runs make photo_runs failed too."""
    f = OUT / f'node/part_a_{enc}.json'
    if not f.exists():
        return None
    doc = jload(f)
    done = doc['done']
    c = sum(x.startswith('cstar__') for x in done)
    p = sum(x.startswith('xphoto__') for x in done)
    n = len(BP.SEEDS)
    # a failed X-c* run means no X photo model (it starts from X-c*'s checkpoint): photo_runs failed too
    cs = 'ok' if c == n else 'failed'
    ph = 'ok' if p == n and cs == 'ok' else 'failed'
    assert doc['status'] == 'failed' or (cs, ph) == ('ok', 'ok'), (enc, doc['status'], c, p)
    return cs, ph


def stage2_state():
    f = OUT / 'stage2/stage2_status.json'
    return jload(f)['status'] if f.exists() else None


def build_status(close_pending=False, strict=True):
    """Write RECORDS/status.json when every gate, every part a result and the stage 2 result is in; else say what is missing
    (strict: raise). Values: 'ok' | 'failed' (bs_outcome.py STATUS SCHEMA). A stage 0 gate that both paths of an encoder
    depend on is written failed under both; failed X-c* runs make photo_runs failed too."""
    gates = load_gates()
    bad, pending, missing = gate_state(gates)
    if pending and close_pending:
        for gid in pending:
            gates[gid]['resolved'] = 'failed'
        jdump(gate_file(), gates)
        bad, pending, missing = gate_state(gates)
    todo = [f'gate {m}' for m in missing] + [f'pending branch {p}' for p in pending]
    enc_doc = {}
    for enc in ENCS:
        st = part_a_state(enc)
        if st is None:
            todo.append(f'part a result {enc}')
            continue
        enc_doc[enc] = dict(cstar_runs=st[0], photo_runs=st[1],
                            stage0={p: 'failed' if f'{enc}:{p}' in bad else 'ok' for p in PATHS})
    s2 = stage2_state()
    if s2 is None:
        todo.append('stage 2 result')
    if todo:
        msg = 'status.json not written; missing: ' + ', '.join(todo)
        if strict:
            raise SystemExit(msg)
        print('BS_RUN ' + msg, flush=True)
        return 0
    status = dict(encoders=enc_doc, part_c=dict(stage0='failed' if 'part_c' in bad else 'ok', stage2=s2))
    jdump(RECORDS / 'status.json', status)
    print(f'BS_RUN status.json written: {json.dumps(status)}', flush=True)
    return 0


def read_status():
    f = RECORDS / 'status.json'
    assert f.exists(), f'{f} missing: run `bs_run.py status` after stage 0, the part a runs and stage 2'
    return jload(f)


def part_alive(enc, status=None):
    """{'paintings': bool, 'photos': bool}: the path is not failed. Before status.json exists (stage 1), from the gates and
    the part a result alone; a missing result counts as alive only when `status` is None and nothing failed."""
    bad = gate_state(load_gates())[0]
    st = part_a_state(enc)
    cs, ph = st if st else ('failed', 'failed')
    return dict(paintings=cs == 'ok' and f'{enc}:paintings' not in bad, photos=ph == 'ok' and f'{enc}:photos' not in bad)


# ============================================================================================== exports (final review F1)
STAR_ARM = 'k11=40'                                  # c*'s arm name (bs_cstar.STAR_ARM): the held exports' folder
SPLITS = ('val', 'test')
EXPORT_HINT = 'run the exports step: `bash bs_run_cpu.sh exports-qwen` / `bash bs_run_cpu.sh exports-part-a --enc <enc>`'


def held_files(enc, seeds=P_SEEDS):
    return [OUT / 'cstar' / enc / 'held' / STAR_ARM / f'{a}__s{sd}.npz' for sd in seeds for a in BP.ASPECTS]


def photo_files(dataset, arm, splits=SPLITS, seeds=P_SEEDS):
    return [OUT / 'exports' / dataset / arm / f'{sp}__s{sd}.npz' for sp in splits for sd in seeds]


def lacking(files):
    """The files (or their sidecars) that do not exist."""
    return [f for f in files if not (f.is_file() and f.with_suffix('.json').is_file())]


def export_valid(npz, kind):
    """The export file and its sidecar exist and agree: photo/EmoSet exports (bs_photo): sidecar `sha256` is the file's sha256;
    held exports (bs_cstar, rb_student.write_outputs): each array's sha256 in the sidecar equals the stored array's."""
    npz = Path(npz)
    side = npz.with_suffix('.json')
    if not (npz.is_file() and side.is_file()):
        return False
    try:
        info = jload(side)
        if kind == 'photo':
            return info.get('sha256') == hashlib.sha256(npz.read_bytes()).hexdigest()
        import numpy as np
        with np.load(npz, allow_pickle=False) as z:
            return sorted(z.files) == sorted(info['arrays']) and all(
                hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() == v['sha256'] for k, v in info['arrays'].items())
    except (OSError, ValueError, KeyError):             # a broken file or sidecar is re-exported (the CLI asserts on other content)
        return False


def export_job(item, seeds):
    """(step name, argv, files written, kind) of one export item: ('held', enc) or ('photo', dataset, arm)."""
    if item[0] == 'held':
        enc = item[1]
        return (f'held_{enc}', py('bs_cstar', 'export', '--encoder', enc, '--kind', 'held', '--seeds', *seeds, '--device', 'cpu'),
                held_files(enc, seeds), 'held')
    _, dataset, arm = item          # bs_photo export runs on CPU only (bs_photo.infer), so it has no --device
    return (f'{dataset}_{arm}', py('bs_photo', 'export', '--arm', arm, '--dataset', dataset, '--splits', *SPLITS, '--seeds', *seeds),
            photo_files(dataset, arm, SPLITS, seeds), 'photo')


def run_exports(a):
    """Local CPU export steps (final review F1): `--set qwen` (EmoSet exports of photo_qwen and cstar_qwen) or `--set part-a
    --enc E ...` (X-c* held export, X photo exports on Affection and EmoSet). Idempotent: a job whose files all exist with a
    matching sha256 is skipped; the CLIs also refuse other content under an existing name."""
    seeds = tuple(a.seeds or P_SEEDS)
    items, notes = [], []
    if a.set == 'qwen':
        assert not a.enc, '--enc belongs to --set part-a'
        items = [('photo', 'emoset', 'photo_qwen'), ('photo', 'emoset', 'cstar_qwen')]
    else:
        assert a.enc, '--set part-a needs --enc E [E ...]'
        for enc in a.enc:
            st = part_a_state(enc)
            if st is None:
                raise SystemExit(f'REFUSED: no part a result OUT/node/part_a_{enc}.json: pull the node run first')
            if st[0] == 'ok':
                items.append(('held', enc))
            else:
                notes.append(f'{enc}: X-c* runs failed: no held export (paintings path failed, D18)')
            if st[1] == 'ok':
                items += [('photo', 'affection', f'photo_{enc}'), ('photo', 'emoset', f'photo_{enc}')]
            else:
                notes.append(f'{enc}: X photo runs failed: no photo_{enc} exports (photos path failed, D18)')
    for n in notes:
        print(f'BS_RUN exports SKIPPED {n}', flush=True)
    bad, left = 0, 0
    for item in items:
        name, argv, files, kind = export_job(item, seeds)
        if all(export_valid(f, kind) for f in files):
            print(f'BS_RUN exports {name}: {len(files)} files present with matching sha256, skipped', flush=True)
            continue
        bad += step('exports', name, argv)['rc'] != 0
        left += 0 if STUB else len(lacking(files))
    print(f'BS_RUN exports {a.set}: {f"{bad} step(s) FAILED" if bad else "done"}; files still missing: {left}'
          + (f'; skipped arms: {len(notes)}' if notes else ''), flush=True)
    return 1 if bad or left else 0


def require_exports(files, what):
    miss = lacking(files)
    if miss:
        raise SystemExit(f'REFUSED: {what}: {len(miss)} export file(s) missing, e.g. {miss[0]}; {EXPORT_HINT}')


# ============================================================================================== stage 1 and 2
def stage1_record_path(enc):
    return RECORDS / f'stage1_{enc}.json'


def stage1_sources(enc):
    s = OUT / 'stage1'
    return dict(artclu=s / f'artclu_ranges_{enc}.json', artret_ranges=s / f'artret_ranges_{enc}.json',
                artret_picks=s / f'artret_picks_{enc}.json', affection=s / f'affection_ranges_{enc}.json')


def run_stage1(a):
    done = []
    for enc in a.enc or ENCS:
        alive = part_alive(enc)
        if not (alive['paintings'] or alive['photos']):
            print(f'BS_RUN stage1 {enc}: both paths failed or runs missing, skipped (no verdict, D18)', flush=True)
            continue
        if alive['photos']:
            require_exports(photo_files('affection', f'photo_{enc}', ('val',)), f'stage 1 {enc}: Affection validation exports')
        ok = True
        if alive['paintings']:
            for nm, argv in (('artclu_ranges', py('bs_artclu', 'ranges', '--enc', enc)),
                             ('artret_ranges', py('bs_artret', 'ranges', '--enc', enc)),
                             ('artret_pick', py('bs_artret', 'pick', '--enc', enc))):
                ok &= step(1, f'{nm}_{enc}', argv)['rc'] == 0
        if alive['photos']:
            ok &= step(1, f'affection_ranges_{enc}', py('bs_affection', 'ranges', '--enc', enc))['rc'] == 0
        if not ok:
            print(f'BS_RUN stage1 {enc}: a step failed; no record written (rerun, or fix, then rerun)', flush=True)
            return 1
        src = stage1_sources(enc)
        keys = (['artclu', 'artret_ranges', 'artret_picks'] if alive['paintings'] else []) + (['affection'] if alive['photos'] else [])
        rec = dict(encoder=enc, kind='stage1_record', written=now(), **{k: jload(src[k]) for k in keys})
        f = stage1_record_path(enc)
        text = json.dumps(rec, indent=1, allow_nan=False) + '\n'
        if f.exists():
            old = jload(f)
            old.pop('written', None)
            new = json.loads(text)
            new.pop('written', None)
            assert old == new, f'{f} holds other content: picks are frozen once written'
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        commit([f], f'r14 stage 1: seed ranges and read-out picks of {enc} (before any held number)')
        done.append(enc)
    print(f'BS_RUN stage1 records: {done}', flush=True)
    return 0


def check_stage1_record(enc, paths):
    """The held-data guard of ArtELingo and Affection scoring: the record is committed and equals the stage 1 files the
    scorers read (a scorer reads OUT/stage1, so a changed file would bypass the record)."""
    f = stage1_record_path(enc)
    require(f, f'stage 3 {enc}: stage 1 record')
    rec, src = jload(f), stage1_sources(enc)
    for key in (('artclu', 'artret_ranges', 'artret_picks') if 'paintings' in paths else ()) + (('affection',) if 'photos' in paths else ()):
        if not (key in rec and rec[key] == jload(src[key])):
            raise SystemExit(f'REFUSED: {src[key]} differs from {f}[{key}]')


def check_stage2_ready():
    """Stage 2 runs after part a and its exports (final review F3): `descriptive_available` must see the X photo models'
    EmoSet exports, and the record is frozen once written. For each encoder the part a result exists and either its
    photo_<enc> EmoSet exports exist or its photo runs failed; the Qwen EmoSet exports (photo_qwen, cstar_qwen) exist."""
    for enc in ENCS:
        st = part_a_state(enc)
        if st is None:
            raise SystemExit(f'REFUSED: stage 2 runs after part a: no OUT/node/part_a_{enc}.json (pull the node run first)')
        if st[1] != 'failed':
            require_exports(photo_files('emoset', f'photo_{enc}'), f'stage 2: photo_{enc} EmoSet exports')
    for arm in ('photo_qwen', 'cstar_qwen'):
        require_exports(photo_files('emoset', arm), f'stage 2: {arm} EmoSet exports')


def run_stage2(a):
    for enc in ENCS:                                    # also when part c is stopped: the status needs every part a result
        if part_a_state(enc) is None:
            raise SystemExit(f'REFUSED: stage 2 runs after part a: no OUT/node/part_a_{enc}.json (pull the node run first)')
    if load_gates().get('xi') is None:
        print('BS_RUN stage2: gate 0(xi) not run yet', flush=True)
    bad = gate_state(load_gates())[0]
    st = OUT / 'stage2/stage2_status.json'
    if 'part_c' in bad:
        jdump(st, dict(status='failed', why=f'stage 0 gates failed: {bad["part_c"]}', at=now()))
        print('BS_RUN stage2: part c stopped by stage 0 (no verdict, §6)', flush=True)
        return build_status(strict=False)
    check_stage2_ready()
    r = step(2, 'stage2', py('bs_emoset_score', 'stage2'))
    src, dst = OUT / 'stage2/stage2_emoset.json', RECORDS / 'stage2_emoset.json'
    if r['rc'] or not src.exists():
        jdump(st, dict(status='failed', why=f'stage2 step rc={r["rc"]}', at=now()))
        return build_status(strict=False) or 1
    if dst.exists():
        assert dst.read_bytes() == src.read_bytes(), f'{dst} holds another record: picks are frozen once written'
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    commit([dst], 'r14 stage 2: EmoSet r_E, read-out and wording picks (before any EmoSet test number)')
    jdump(st, dict(status='ok', at=now()))
    return build_status(strict=False)


# ============================================================================================== stage 3
def run_stage3(a):
    status = read_status()                              # raises if stage 0, part a or stage 2 are not all in
    commit([RECORDS / 'status.json'], 'r14 stage 3: status.json (stage 0 gates, part a runs, stage 2)')
    extra = {'n_clu': ['--n-clu', a.n_clu] if a.n_clu else [], 'n_ret': ['--n-ret', a.n_ret] if a.n_ret else []}
    draws_clu = ['--n-draws', a.n_clu] if a.n_clu else []
    draws_ret = ['--n-draws', a.n_ret] if a.n_ret else []
    bad = 0
    for enc in a.enc or ENCS:
        s = status['encoders'][enc]
        alive = [p for p in PATHS if s['stage0'][p] == 'ok' and s['cstar_runs' if p == 'paintings' else 'photo_runs'] == 'ok']
        if not alive:
            print(f'BS_RUN stage3 {enc}: no path alive (inconclusive, D18)', flush=True)
            continue
        check_stage1_record(enc, alive)
        if 'paintings' in alive:
            require_exports(held_files(enc), f'stage 3 {enc}: X-c* held exports')
        if 'photos' in alive:
            require_exports(photo_files('affection', f'photo_{enc}'), f'stage 3 {enc}: Affection exports')
        if 'paintings' in alive:
            bad += step(3, f'artclu_score_{enc}', py('bs_artclu', *draws_clu, 'score', '--enc', enc, '--loader', 'bs_stores'))['rc'] != 0
            bad += step(3, f'artret_score_{enc}', py('bs_artret', *draws_ret, 'score', '--enc', enc))['rc'] != 0
        if 'photos' in alive:
            bad += step(3, f'affection_score_{enc}', py('bs_affection', *extra['n_clu'], *extra['n_ret'], 'score', '--enc', enc))['rc'] != 0
    pc = status['part_c']
    if pc['stage0'] == 'ok' and pc['stage2'] == 'ok':
        require(RECORDS / 'stage2_emoset.json', 'stage 3 part c: stage 2 record')
        bad += step(3, 'emoset_stage3', py('bs_emoset_score', 'stage3', *extra['n_clu'], *extra['n_ret']))['rc'] != 0
    else:
        print(f'BS_RUN stage3 part c skipped: {pc} (inconclusive, D18)', flush=True)
    if bad:
        print(f'BS_RUN stage3: {bad} scoring step(s) failed; cells incomplete', flush=True)
        return 1
    sha = c13_hashes()
    jdump(OUT / 'stage3/c13_sha256.json', sha)
    for k, v in sha.items():
        print(f'BS_RUN stage3 sha256 {k} {v}', flush=True)
    print(f'BS_RUN stage3: cells written ({len([k for k in sha if k.startswith("stage3/")])} files); NO verdict yet (C13: Codex re-derives, '
          f'then RECORDS/c13.json is committed, then `bs_run.py decide`)', flush=True)
    return 0


def c13_hashes():
    """{'stage3/<cells file>': sha256, 'records/<stage 1 or 2 record>': sha256}: the files C13's re-derivation covers (§3.4:
    the cells of OUT/stage3 and the committed stage 1 and stage 2 records)."""
    out = {f'stage3/{f.name}': hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((OUT / 'stage3').glob('cells_*.json'))}
    for f in sorted(RECORDS.glob('stage1_*.json')) + [x for x in (RECORDS / 'stage2_emoset.json',) if x.exists()]:
        require(f, 'decide: stage 1 / stage 2 record')
        out[f'records/{f.name}'] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def run_decide(a):
    """Verdicts, after C13 (§3.4: Codex re-derives the cells "before any verdict is written"). Refuses unless RECORDS/c13.json
    is committed in git HEAD with "pass": true and a "sha256" map equal to c13_hashes() (the current cells files and the
    committed stage 1 and stage 2 records). Format of RECORDS/c13.json, written by the run chat after Codex's re-derivation:
        {"pass": true, "sha256": {"stage3/cells_affection.json": "<hex>", ..., "records/stage1_siglip2.json": "<hex>", ...},
         "by": "...", "at": "...", "notes": "..."}
    (`stage3` prints the map and writes it to OUT/stage3/c13_sha256.json.) Then bs_outcome decide -> OUT/stage3/outcomes.json
    and RECORDS/verdicts.json (committed)."""
    f = RECORDS / 'c13.json'
    require(f, 'decide: C13 re-derivation record (the run chat commits it after Codex re-derived the cells)')
    rec = jload(f)
    if rec.get('pass') is not True:
        raise SystemExit(f'REFUSED: {f}: "pass" is not true: C13 did not pass; no verdict')
    want, have = rec.get('sha256'), c13_hashes()
    if want != have:
        diff = sorted(k for k in set(want or {}) | set(have) if (want or {}).get(k) != have.get(k))
        raise SystemExit(f'REFUSED: {f}: "sha256" differs from the current files: {diff}; C13 covered other content')
    dec = ['decide', '--status', RECORDS / 'status.json', '--nearcopy', RECORDS / 'nearcopy_emoset.json'] + (['--smoke'] if a.smoke else [])
    r = step(3, 'decide', py('bs_outcome', *dec))
    out = OUT / 'stage3/outcomes.json'
    if r['rc'] or not out.exists():
        return 1
    shutil.copyfile(out, RECORDS / 'verdicts.json')
    commit([RECORDS / 'verdicts.json'], 'r14 stage 3: verdicts (bs_outcome decide, after C13)')
    return 0


# ============================================================================================== stages 4, 5, standin
def run_stage4(a):
    bad = 0
    for nm in ('knob', 'photo', 'ablation'):
        extra = ['--smoke'] if (nm == 'ablation' and a.smoke) else []
        bad += step(4, f'partb_{nm}', py('bs_partb', nm, *extra, *(['--n-clu', a.n_clu] if a.n_clu else []),
                                          *(['--n-ret', a.n_ret] if a.n_ret else [])))['rc'] != 0
    d = OUT / 'stage4'
    names = ['partb_knob', 'partb_photo', 'partb_ablation' + ('_smoke' if a.smoke else '')]
    missing = [n for n in names if not (d / f'{n}.json').exists()]
    jdump(d / 'stage4_status.json', dict(ok=not bad and not missing, missing=missing, at=now()))
    print(f'BS_RUN stage4: sheets missing {missing}' if missing else 'BS_RUN stage4: three sheets written', flush=True)
    return 1 if bad or missing else 0


def run_diag(a, stage=5):
    import bs_diag                                      # ticket 12b; lazy so this module works without it
    t0 = time.time()
    (OUT / f'stage{stage}').mkdir(parents=True, exist_ok=True)
    bs_diag.run_all(smoke=a.smoke)
    print(f'BS_RUN stage{stage} diag seconds={time.time() - t0:.1f}', flush=True)
    return 0


def run_standin(a):
    assert Path(OUT) != Path(BP.DEFAULT_OUT), 'the Qwen stand-in needs its own BS_OUTPUTS (its cells must not reach production)'
    bad = 0
    for nm, argv in (('affection_prepare', py('bs_affection', 'standin', 'prepare')),
                     ('artclu_ranges', py('bs_artclu', 'ranges', '--enc', 'qwen')),
                     ('artclu_score', py('bs_artclu', 'score', '--enc', 'qwen', '--loader', 'bs_stores')),
                     ('artret_ranges', py('bs_artret', 'ranges', '--enc', 'qwen')),
                     ('artret_pick', py('bs_artret', 'pick', '--enc', 'qwen')),
                     ('artret_score', py('bs_artret', 'score', '--enc', 'qwen')),
                     ('affection_ranges', py('bs_affection', 'ranges', '--enc', 'qwen')),
                     ('affection_score', py('bs_affection', 'score', '--enc', 'qwen')),
                     ('affection_check', py('bs_affection', 'standin', 'check'))):
        bad += step(3, f'standin_{nm}', argv)['rc'] != 0
    return 1 if bad else 0


# ============================================================================================== CLI
def main(argv=None):
    global NO_COMMIT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-commit', action='store_true', help='same as BS_RUN_NO_COMMIT=1')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s0 = sub.add_parser('stage0')
    s0.add_argument('--parts', choices=('cpu', 'gpu', 'all'), default='all')
    s0.add_argument('--only', nargs='+')
    s0.add_argument('--resolve', nargs='+', metavar='ID=ok|failed')
    s0.add_argument('--close-pending', action='store_true')
    st = sub.add_parser('status')
    st.add_argument('--close-pending', action='store_true')
    s1 = sub.add_parser('stage1')
    s1.add_argument('--enc', nargs='+', choices=ENCS)
    sub.add_parser('stage2')
    ex = sub.add_parser('exports')
    ex.add_argument('--set', required=True, choices=('qwen', 'part-a'))
    ex.add_argument('--enc', nargs='+', choices=ENCS)
    ex.add_argument('--seeds', type=int, nargs='+', choices=P_SEEDS, help='default: 0 to 4 (fewer only for a dry check)')
    s3 = sub.add_parser('stage3')
    s3.add_argument('--enc', nargs='+', choices=ENCS)
    s3.add_argument('--n-clu', type=int)
    s3.add_argument('--n-ret', type=int)
    dc = sub.add_parser('decide')
    dc.add_argument('--smoke', action='store_true', help='draw counts other than the protocol\'s (outcome code allows them)')
    s4 = sub.add_parser('stage4')
    s4.add_argument('--n-clu', type=int)
    s4.add_argument('--n-ret', type=int)
    s4.add_argument('--smoke', action='store_true')
    for nm in ('diag', 'stage5'):
        sub.add_parser(nm).add_argument('--smoke', action='store_true')
    sub.add_parser('standin')
    a = ap.parse_args(argv)
    if a.no_commit:
        NO_COMMIT = True
    t0 = time.time()
    fn = {'stage0': run_stage0, 'status': lambda a: build_status(a.close_pending), 'stage1': run_stage1, 'stage2': run_stage2,
          'exports': run_exports, 'stage3': run_stage3, 'decide': run_decide, 'stage4': run_stage4, 'diag': lambda a: run_diag(a, 5), 'stage5': lambda a: run_diag(a, 5),
          'standin': run_standin}[a.cmd]
    rc = fn(a) or 0
    print(f'BS_RUN {a.cmd} done rc={rc} seconds={time.time() - t0:.1f}', flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
