"""Sample-scale trainer of the recipe sweep: any `rs_sheet` configuration trained as 12-27's GS-0.02 (model G) with the
knob layer's patches (`rs_knobs`); exports of the three image heads on P from each head's best state. `regress v1a`
is V1a's training half.

G enters no patch: `run_config` reaches `rg_train.train_cohort` (with `rg_loss.switched_head_loss` and
`ms_train.run` under it), `Data` and `load_graphs` exactly as `rg_train.run_variant` does.
"""
import argparse
import hashlib
import json
import subprocess
import socket
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

import rs_paths as paths
import rs_sheet as S
import rs_knobs
import ab_train
import eg_train3 as eg
import ab_rows
import rb_part1
import rb_rows
import rg_loss
import rg_train
import rg_variants as V
from ab_paths import ms_rows, ms_train

ASPECTS = tuple(paths.ASPECTS)
G_ARM = paths.G_ARM
SEEDS = (0, 1, 2, 3, 4)
MACHINES = ('local', 'node402')
V1A = 'G-v1a'                 # V1a's replays of G, named apart from G's own runs (moved-aside folders)
REPEAT = 'G-repeat'           # G₀′ (rule N3): G's configuration repeated in its own folder
ROUND_FILES = ('rs_train.py', 'rs_knobs.py', 'rs_sheet.py', 'rs_paths.py')
# What decides a run's numbers: the knob layer, the sheet, the 12-27 trainer and loss, and the frozen modules under them.
# A finished run is resumed only if these are unchanged (the CLI and the gate code may change).
TRAINING_FILES = (Path(rs_knobs.__file__), Path(S.__file__), Path(rg_train.__file__), Path(rg_loss.__file__),
                  Path(V.__file__), Path(ms_train.__file__), Path(ms_train.itp_train.__file__),
                  Path(ab_train.__file__), Path(eg.__file__), Path(rb_rows.__file__), Path(ab_rows.__file__),
                  Path(ms_rows.__file__), Path(ms_train.head.__file__), Path(ms_train.ags_trainimg.__file__))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha256(*, training_only=False):
    """SHA-256 of the files a run depends on, by file name (the round's four files too, unless `training_only`)."""
    files = list(TRAINING_FILES)
    if not training_only:
        files += [paths.ROUND / name for name in ROUND_FILES]
    result = {p.name: sha(p) for p in files}
    assert len(result) == len(set(files)), 'two files with one name'
    return result


def jsonable(cfg):
    """The configuration as run.json stores it (the removals as a list)."""
    return dict(cfg, remove=list(cfg['remove']))


def code_commit():
    """HEAD, and whether any Python file of this round or of 12-27 is modified or untracked (rg_train's own check
    sees 12-27's folder only)."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=paths.ROUND, capture_output=True, text=True, check=True).stdout.strip()
    return git('rev-parse', 'HEAD'), bool(git('status', '--porcelain', '--', ':/src/test/20261027_recipe_sweep/*.py',
                                               ':/src/test/20261027_relational_genre/*.py'))


def run_config(data, cfg, seed, outputs, device, *, run_dir=None, smoke=False, max_epochs=None, machine='local',
               name=None):
    """Train `cfg` at `seed` into run_dir(outputs, name, seed) and return its run.json record.

    The structure is `rg_train.run_variant`'s (copied from it: its seeds stop at 3, its folder and record are keyed
    by the arm), with the knob patches entered around it and the record keyed by the configuration's name. Existing
    finished runs are verified and skipped. A run that fails V1b (rg_train's VA3) is rerun once, the first moved
    aside (V1d); a run that stops on a non-finite loss is `failed` and not rerun. `name` overrides the folder and
    record name of the same configuration (G's repeat on the node is 'G-repeat'); `machine` is written as given."""
    name = name or S.name(cfg)
    assert name == S.name(cfg) or (name in (REPEAT, V1A) and cfg == S.config()), (name, S.name(cfg))
    assert machine in MACHINES, machine
    assert seed in SEEDS, seed
    outputs, device = paths.check_outputs(outputs, smoke), rg_train.check_device(device)
    paths.check_imports()
    run_dir = Path(run_dir).resolve() if run_dir is not None else paths.run_dir(outputs, name, seed)
    assert run_dir.is_relative_to(outputs), 'run folder must be under --outputs'
    max_epochs = (2 if smoke else ms_train.CONFIG['max_epochs']) if max_epochs is None else max_epochs
    assert not smoke or max_epochs == 2, '--smoke requires 2 epochs'
    assert 1 <= max_epochs <= ms_train.CONFIG['max_epochs']
    if cfg['remove'] and not smoke:
        # An ablation arm is a removal from the final model F (c* under PICK, else G), trained at seeds 1 to 3 (rule §2.5).
        # The check is `rs_full.require_stage3`, the one the full-scale arms pass (imported here: rs_full imports this module).
        assert seed in S.DEV_SEEDS, f'{name}: the ablation trains at seeds {S.DEV_SEEDS} (seeds 1 to 3), not {seed}'
        import rs_full
        rs_full.require_stage3(outputs, cfg)
    with ExitStack() as stack:
        knobs = rs_knobs.patches(cfg, stack, scale='sample', outputs=outputs)
        s, inputs, p, p_sha = data.get(export_p=True)
        recipe = rg_train.resolved(seed)
        graphs, manifests = rg_train.load_graphs(s, recipe)
        identity = rg_train.input_identity(data, p_sha, manifests)
        switches = V.switches(G_ARM)         # with k3, k4 patched: the genre switches of this run
        same = (name, seed, jsonable(cfg), smoke, max_epochs)
        rerun_of = None
        while True:
            record_path = run_dir / 'run.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
                assert 'name' in record, f'{record_path}: no name (interrupted before its fields were added); move it aside'
                assert (record['name'], record['seed'], record['config'], record['smoke'], record['max_epochs']) == same, \
                    f'{record_path}: configuration mismatch'
                assert record['p_sha256'] == p_sha and record['input_hashes'] == identity, 'run inputs changed'
                assert record['switches'] == switches and record['status'] in ('ok', 'failed')
                have = record['code_sha256']
                assert all(have.get(k) == v for k, v in code_sha256(training_only=True).items()), \
                    'run code_sha256 changed; move it aside and rerun'
                if record['status'] == 'ok' or record['failure_kind'] != 'va3' or 'rerun_of' in record:
                    if record['status'] == 'ok':
                        rg_train.verify_cached_run(run_dir, record, export_p=True)
                    print(f'{name} s{seed}: {record["status"]} run exists; skip', flush=True)
                    return record
                assert not rg_train.va3_failures_aside(outputs, name, seed), 'already rerun once after V1b'
                head, dirty = code_commit()
                assert record['code_commit'] == head and not record['code_dirty'] and not dirty, \
                    'a V1b rerun needs the same clean code'
                rerun_of = rg_train.move_aside(outputs, run_dir, name, seed)
            else:
                # An interrupted rerun keeps its origin, even if its folder is absent or empty.
                previous = rg_train.va3_failures_aside(outputs, name, seed)
                assert len(previous) <= 1, 'already rerun once after V1b'
                if previous:
                    origin = json.loads((previous[0] / 'run.json').read_text())
                    assert (origin['code_commit'], origin['code_dirty']) == code_commit() \
                        and not origin['code_dirty']
                    assert (origin['name'], origin['seed'], origin['config'], origin['smoke'],
                            origin['max_epochs']) == same, 'interrupted V1b rerun configuration changed'
                    assert origin['input_hashes'] == identity, 'interrupted V1b rerun inputs changed'
                    assert all(origin['code_sha256'].get(k) == v for k, v in code_sha256(training_only=True).items()), \
                        'interrupted V1b rerun code_sha256 changed'
                    rerun_of = previous[0]
                if run_dir.exists() and any(run_dir.iterdir()):
                    rg_train.move_aside(outputs, run_dir, name, seed)
            record = rg_train.train_cohort(s, {a: inputs for a in ASPECTS}, graphs, manifests, recipe, G_ARM, seed,
                                           device, run_dir, p=p, p_sha256=p_sha, smoke=smoke, max_epochs=max_epochs,
                                           rerun_of=rerun_of, input_hashes=identity)
            assert record['switches'] == switches
            head, dirty = code_commit()
            record.update(code_commit=head, code_dirty=dirty, cause=record.get('cause'),
                          variant=name, name=name, base_arm=G_ARM, config=jsonable(cfg), knobs=knobs,
                          code_sha256=code_sha256(), machine=machine, host=socket.gethostname())
            eg.atomic_record(run_dir / 'run.json', record)
            if record['status'] == 'failed' and record['failure_kind'] == 'va3' and rerun_of is None:
                continue
            return record


def regress_v1a(data, outputs, device, *, smoke=False, machine='local'):
    """V1a, training half (rule §4.2): G at seeds 0 and 1 gives, for each head, exports on P (and on the old 5,000)
    and a best state equal bit for bit to 12-27's GS-0.02 at the same seeds. P's arrays decide, with the states and
    the old-5,000 arrays also required here."""
    outputs = paths.check_outputs(outputs, smoke)
    items, hashes = {}, {}
    for seed in (0, 1):
        folder = outputs / 'regression' / f'v1a_s{seed}'
        run = run_config(data, S.config(), seed, outputs, device, run_dir=folder, smoke=smoke, machine=machine,
                         name=V1A)
        ref = paths.RG_TRAIN / G_ARM / f's{seed}'
        frozen = json.loads((ref / 'run.json').read_text())
        assert frozen['status'] == 'ok' and frozen['variant'] == G_ARM and frozen['seed'] == seed
        assert frozen['p_sha256'] == paths.P_SHA256
        aspects = {}
        if run['status'] == 'ok':
            for aspect in ASPECTS:
                comparisons = {}
                for label, n in (('P', paths.N_P), ('old5000', paths.N_OLD)):
                    mine = rg_train.read_export(folder / f'{aspect}__{label}.npz', n)
                    old = rg_train.read_export(ref / f'{aspect}__{label}.npz', n)
                    comparisons[f'{label}_img_equal'] = rg_train.bits_equal(mine, old)
                    hashes[f's{seed}/{aspect}/{label}'] = dict(replay=rb_part1.array_sha256(mine),
                                                              reference=rb_part1.array_sha256(old))
                mine_state, old_state = folder / f'{aspect}.pt', ref / f'{aspect}.pt'
                comparisons['state_equal'] = rg_train.states_equal(mine_state, old_state)
                hashes[f's{seed}/{aspect}/state'] = dict(replay=sha(mine_state), reference=sha(old_state))
                aspects[aspect] = comparisons
        passed = run['status'] == 'ok' and len(aspects) == len(ASPECTS) and all(
            all(c.values()) for c in aspects.values())
        items[f's{seed}'] = dict(aspects=aspects, run_status=run['status'], run_cause=run.get('cause'),
                                 reference=str(ref), run_dir=str(folder), **{'pass': passed})
    commit, dirty = code_commit()
    record = dict(items=items, smoke=smoke, p_sha256=paths.P_SHA256, code_commit=commit, code_dirty=dirty,
                  code_sha256=code_sha256(), rg_code_sha256=rg_train.rg_code_sha256(), input_hashes=hashes)
    record['pass'] = all(i['pass'] for i in items.values()) and not smoke
    eg.atomic_record(outputs / 'gates/v1a_train.json', record)
    return record


@threadpool_limits.wrap(limits=8)
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'regress'):
        cmd = sub.add_parser(command)
        cmd.add_argument('--outputs', type=Path, default=None,
                         help=f'default {paths.OUTPUTS}; with --smoke {paths.SMOKE_ROOT}/r10-rs-smoke')
        cmd.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
        cmd.add_argument('--machine', choices=MACHINES, default='local', help='written to run.json as given')
        cmd.add_argument('--smoke', action='store_true', help='scratch outputs only; 5000 fit pairs, 2 epochs')
        if command == 'train':
            cmd.add_argument('--configs', nargs='+', required=True, help="run names, e.g. G k6=0.0005 k2=8+k6=0.0005; G-repeat is G in its own folder")
            cmd.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=[0])
        else:
            cmd.add_argument('gate', choices=['v1a'])
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    outputs = args.outputs or (paths.SMOKE_ROOT / 'r10-rs-smoke' if args.smoke else paths.OUTPUTS)
    outputs = paths.check_outputs(outputs, args.smoke)
    data = rg_train.Data()
    if args.command == 'train':
        configs = [(text, S.config() if text == REPEAT else S.parse(text)) for text in args.configs]
        for name, cfg in configs:
            for seed in args.seeds:
                record = run_config(data, cfg, seed, outputs, args.device, smoke=args.smoke, machine=args.machine,
                                    name=name)
                print(f'RS_TRAIN {name} s{seed} {record["status"]} {record.get("cause", "")}'.rstrip(), flush=True)
        return 0      # a failed run is a recorded outcome (rule V1d), not an error of the queue
    record = regress_v1a(data, outputs, args.device, smoke=args.smoke, machine=args.machine)
    print(f'RS_TRAIN V1A pass={record["pass"]}', flush=True)
    return 0 if record['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
