"""CPU-only reproducible runner smoke. Run ONE stage at a time: pp_run_smoke.py 0|1|2|3|4.

Requires PP_SMOKE=1 and PP_OUTPUTS. Inputs, exports and timings are explicitly synthetic;
MMArt/SemArt statistics, matched export reader, merge, outcome checks and cost arithmetic are real.
Uses the existing scorer test worlds to avoid maintaining a second synthetic data implementation.
Leaves every artifact for ticket 13; reruns update deterministic files, never remove a folder.
"""
import pp_paths as PP
import pp_register  # noqa: F401
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pp_run as R

ENCS = ('clip_b32', 'clip_l14')


def mmart_world():
    import test_pp_mmart as fixture
    import mt_data
    import mt_paths
    fixture.N_VAL, fixture.N_TEST, fixture.N_EXTRA = 64, 80, 0
    w = fixture.World(PP.OUT, np.random.default_rng(7))
    mt_data.keys = lambda name: list({'val': w.val, 'test': w.test, 'new': w.new}[name])
    mt_data.labels = lambda keys, a: np.asarray([w.label[a][k] for k in keys], dtype=np.int64)
    mt_data.features = lambda view, keys: w.view(view, keys)
    mt_paths.CSTAR_ROOT['clip_b32'] = PP.OUT / 'cstar/clip_b32'
    w.write_exports('clip_b32', PP.OUT / 'mmart')
    import pp_matched as matched
    # The smoke mapper identity is local too; keep the real sidecar/hash/key checks active.
    matched.mapper_weights_sha256 = lambda enc, seed: matched.sha_file(PP.OUT / 'smoke_inputs' / f'mapper_{seed}.pt')
    for s in PP.SEEDS:
        mapper = PP.OUT / 'smoke_inputs' / f'mapper_{s}.pt'
        mapper.parent.mkdir(parents=True, exist_ok=True)
        mapper.write_bytes(f'standin {s}'.encode())
        matched.write_export(PP.OUT, 'mmart_new', 'clip_b32', s, w.new, w.probs('clip_b32', s, w.new),
                             mapper=mapper, mapper_sha256=matched.sha_file(mapper), view='clip_b32')
    return w


def stage0():
    # Missing/reduced gates deliberately remain failed, including when the replay process exited zero.
    for p in ('r14', 'r15', 'semart'):
        R.jdump(PP.OUT / f'stage0/gate_iv_{p}.json', dict(passed=False, values_ok=True, smoke=True))
    R.jdump(PP.OUT / 'stage0/gate_vi.json', {'pass': True, 'smoke': True})
    R.jdump(PP.record_path('semart_set.json'), dict(n_set=80, artists_set=10, smoke=True,
            stores_checked=[str(p.relative_to(PP.SEMART_PRE)) for (ds, _), p in PP.NEW_STORE.items() if ds == 'semart']))
    w = mmart_world()
    R.jdump(PP.OUT / 'smoke_inputs/keys.json', dict(val=w.val, test=w.test, encoders=ENCS,
                                                 draws=PP.SMOKE_DRAWS, standins=True))
    R.write_status()
    assert R.jload(PP.record_path('status.json'))['gates']['iv'] == 'failed'


def stage1():
    import pp_mmart as m
    mmart_world()
    block = m.stage1_block('clip_b32', workers=1, expect_n=None)
    m.write_stage1('clip_b32', block)
    # Plumbing stand-ins for r14's already separately smoke-tested stage1 blocks.
    r14 = R.jload(PP.R14_RECORDS / 'stage1_siglip1.json')
    def rename(x):
        if isinstance(x, dict):
            return {k: rename(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rename(v) for v in x]
        return 'clip_b32' if x == 'siglip1' else x
    prefixes = dict(artclu='artclu_ranges', artret_ranges='artret_ranges', artret_picks='artret_picks', affection='affection_ranges')
    for key, prefix in prefixes.items():
        R.jdump(PP.OUT / 'stage1' / f'{prefix}_clip_b32.json', rename(r14[key]))
    import pp_semart_score as ss
    for enc in ENCS:
        entry = ss.stage1_entry(.011, .0042, dict(cstar='center+k10', frozen='center+k5', teacher='none+k3'),
                               'synthetic validation units for smoke', dict.fromkeys(ss.ARMS, 'standin'))
        R.jdump(PP.OUT / 'stage1' / f'semart_{enc}.json', {enc: entry})
    R.merge_records(ENCS)
    # Demonstrate the real git HEAD refusal against the round record names, without writing records or committing.
    original = PP.read_committed_record
    def strict(name):
        path = PP.RECORDS / name
        rel = path.relative_to(PP.REPO)
        head = subprocess.run(['git', '-C', str(PP.REPO), 'show', f'HEAD:{rel}'], capture_output=True, text=True)
        if head.returncode or head.stdout != PP.record_path(name).read_text():
            raise RuntimeError(f'{rel} is not committed unchanged in HEAD')
        return json.loads(head.stdout)
    PP.read_committed_record = strict
    try:
        refused = R.main(['2', '--enc', *ENCS])
        assert refused == 2
        R.jdump(PP.OUT / 'stage1/refusal.json', dict(stage2_rc=refused, reason='records not committed unchanged in HEAD'))
    finally:
        PP.read_committed_record = original


def stage2():
    import pp_mmart as m
    import mt_clu
    import pp_semart_score as ss
    from test_pp_semart_score import World, FakeInputs
    mmart_world()
    block = m.read_block('clip_b32')
    # t08 trick: bypass full row-count assertions, patch mt_data.keys, keep the real matched_new export route.
    m.write_cells('clu', m.score_clu('clip_b32', block=block, n_draws=PP.SMOKE_DRAWS['clu'], workers=1, expect_n=None))
    m.write_cells('ret', m.score_ret('clip_b32', block=block, n_draws=PP.SMOKE_DRAWS['ret'], workers=1, expect_n=None))
    world = World(n=80, nval=64)
    ss.n_val_rows = lambda: 64
    order = PP.order(world.keys)
    inputs = FakeInputs(world)
    doc = PP.read_committed_record('stage1_semart.json')
    np.savez(PP.OUT / 'smoke_inputs/semart.npz', keys=order, labels=inputs.labels(order),
             frozen=inputs.frozen_eval('clip_b32', order), teacher=inputs.teacher_eval(order),
             **{f'cstar_s{s}': inputs.cstar_eval('clip_b32', s, order) for s in PP.SEEDS})
    for enc in ENCS:
        entry = doc[enc]
        ss.write_cells('semart_clu', enc, ss.score_clu(enc, inputs, order, entry['units']['clu'], PP.SMOKE_DRAWS['clu'], 1))
        ss.write_cells('semart_ret', enc, ss.score_ret(enc, inputs, order, entry['units']['ret'], entry['picks'], PP.SMOKE_DRAWS['ret'], 1))


def stage3():
    import pp_probe as p
    names = {p.default_pass(e): 10. for e in ENCS}
    names.update({p.heads_pass(e): .1 for e in ENCS})
    names.update({name: 70. for name in p.TEACHER_PASSES})
    names['qwen'] = 86.3
    passes = {}
    for name, ms in names.items():
        entry = dict(encoder='qwen' if name.startswith('qwen') else name.replace('heads_', ''),
                     kind='heads' if name.startswith('heads_') else 'model', repeats={}, e2e=None, standin=True)
        for r in range(3):
            rec = p.batch_record(name, entry['kind'], r, [32, 32], [ms * 32 / 1000] * 2,
                                 cpu_load_before=[0, 0, 0], cpu_load_after=[0, 0, 0], smoke=True)
            filename = f'{name}__r{r}.json'
            R.jdump(PP.OUT / 'probe/batches' / filename, rec)
            entry['repeats'][str(r)] = dict(file=filename, ms_per_image=rec['ms_per_image'], total_seconds=rec['total_seconds'])
        passes[name] = entry
    for name, ms in [('teacher', 210.), ('qwen', 86.3), *[(e, 10.) for e in ENCS]]:
        rec = p.batch_record(name, 'e2e', None, [32, 32], [ms * 32 / 1000] * 2,
                             cpu_load_before=[0, 0, 0], cpu_load_after=[0, 0, 0], smoke=True)
        filename = f'{name}__e2e.json'
        R.jdump(PP.OUT / 'probe/batches' / filename, rec)
        passes.setdefault(name, dict(encoder='qwen', kind='model', repeats={}))['e2e'] = dict(
            file=filename, ms_per_image=rec['ms_per_image'], total_seconds=rec['total_seconds'])
    R.jdump(PP.OUT / 'probe/passes.json', dict(meta=dict(n_images=64, batch_size=32, repeats=3, warmup_batches=3, smoke=True),
                                            passes=passes, repeats_discarded=[]))
    assert p.cmd_cells(ENCS) == 0


def stage4():
    import pp_outcome as o
    R.write_status()
    status = PP.read_committed_record('status.json')
    cells, cost, files = o.load_cells(PP.OUT)
    bad = dict(status, extra='wrong schema')
    R.jdump(PP.OUT / 'stage4/wrong_status.json', bad)
    proc = subprocess.run(R.py('pp_outcome', 'decide', '--status', PP.OUT / 'stage4/wrong_status.json', '--smoke'),
                          capture_output=True, text=True)
    (PP.OUT / 'stage4/wrong_status.log').write_text(proc.stdout + proc.stderr)
    assert proc.returncode != 0 and 'top level' in proc.stderr
    R.jdump(PP.OUT / 'stage4/refusal.json', dict(wrong_status_rc=proc.returncode))
    o.ENCODERS = ENCS  # smoke reports exactly the two encoders whose complete cost inputs were written
    res = o.decide(cells, status, cost, counts=dict(mmart_n=80, semart_n=80, semart_artists=10), smoke=True)
    res['cell_files'] = files
    R.jdump(PP.OUT / 'stage4/outcomes.json', res)
    print(o.table(res))
    # These diagnostics consume the saved cells and timings without needing full real stores.
    # Production pp_run always calls the full `pp_diag.py run`.
    kinds = ['gain_share', 'percept_strongest', 'cost_rows', 'node_time']
    units = PP.OUT / 'node/units.jsonl'
    if not units.exists():
        units.parent.mkdir(parents=True, exist_ok=True)
        units.write_text(''.join(json.dumps(dict(kind='standin', enc=e, seed=s, status='ok', seconds=1.,
                                                host='smoke', attempts=1)) + '\n' for e in ENCS for s in PP.SEEDS))
    old_diag = PP.OUT / 'stage4/diag'
    if old_diag.exists():
        old_diag.rename(old_diag.with_name('diag_smoke_previous_' + str(__import__('time').time_ns())))
    rec = R.run_step('stage4', 'diagnostics', R.py('pp_diag', 'run', '--only', *kinds, '--workers', 1))
    R.jdump(PP.OUT / 'stage4/diagnostics_status.json', dict(rc=rec['rc'], failed=rec['failed'],
                                                         module_present=(PP.ROUND / 'pp_diag.py').exists(), kinds=kinds))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['0', '1', '2', '3', '4'])
    ap.add_argument('--worker', action='store_true')
    args = ap.parse_args()
    assert PP.SMOKE and 'PP_OUTPUTS' in __import__('os').environ
    stage = int(args.stage)
    if args.worker:
        globals()[f'stage{stage}']()
        return 0
    rec = R.run_step(f'stage{stage}', f'smoke_{stage}', R.py('pp_run_smoke', stage, '--worker'))
    assert not rec['failed'], f'smoke stage {stage} failed: {rec["log"]}'
    files = sorted(str(p.relative_to(PP.OUT)) for p in (PP.OUT / f'stage{stage}').rglob('*') if p.is_file())
    R.jdump(PP.OUT / f'stage{stage}/smoke_result.json', dict(stage=stage, passed=True, files=files))
    print(f'SMOKE stage{stage} passed; {len(files)} stage files; {PP.OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
