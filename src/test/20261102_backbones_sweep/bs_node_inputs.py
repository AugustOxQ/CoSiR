"""r14 node inputs: what the node jobs read, from where, and the checks around it (ticket 10, part b).

`bs_node_inputs.json` lists the stored inputs a node job reads: `{key, local, node, groups, map_local?}`. `local` is the
path the code reads (bound there inside the job's chroot), `node` is where the cluster CLI puts it on node402 (through
DATA_MAP), `map_local` is the clean-named symlink that DATA_MAP uses when `local` has `=` in it (the CLI refuses those).
The sub-commands other than `probe` use only the standard library (the job script runs them with the node's system
python before the chroot).

    groups-for MODE [--encoder E] [--kind K ...] [GROUP ...]   the groups a script mode needs
    overrides GROUP ...      the Hydra overrides `+bs_in.<key>_path=<node path>` (what `cluster check` composes)
    binds GROUP ...          `NODE_PATH<TAB>LOCAL_PATH` lines (the job script binds each)
    datamap [--conf FILE]    DATA_MAP entries the new node paths need (with --conf: only those missing there)
    verify [GROUP ...]       every local source and every map_local exists (run on this machine)
    probe GROUP ...          in the chroot: every input exists and reads, packages import, CUDA device; writes JSON
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
ITEMS = json.loads((ROUND / 'bs_node_inputs.json').read_text())
NEW_PREFIX = '/local/wding/r14_in/'
ENCODERS = ('siglip2', 'siglip1', 'clip_l14')
KINDS = ('photo', 'knob')
AUX = ('aux_art', 'qwen_art', 'aux_aff', 'qwen_aff')
ALL_GROUPS = sorted({g for it in ITEMS for g in it['groups']})


def groups_for(mode, encoder=None, kinds=(), extra=()):
    if mode == 'probe':
        select(extra)                                            # refuses an unknown group
        return list(extra)
    if mode == 'smoke':
        return list(AUX) + ['art_siglip2', 'aff_siglip2']
    if mode == 'part-a':
        assert encoder in ENCODERS, f'--encoder must be one of {ENCODERS}'
        return list(AUX) + [f'art_{encoder}', f'aff_{encoder}']
    if mode == 'viii':
        return ['aux_art', 'qwen_art', 'viii']
    if mode == 'prep':
        return list(AUX)
    if mode == 'sweep':
        assert kinds and set(kinds) <= set(KINDS), f'--kind must name {KINDS}'
        out = []
        if 'photo' in kinds:
            out += ['aux_aff', 'qwen_aff']
        if 'knob' in kinds:
            out += ['aux_art', 'qwen_art']
        return out
    raise SystemExit(f'unknown mode {mode}')


def select(groups):
    unknown = sorted(set(groups) - set(ALL_GROUPS))
    assert not unknown, f'unknown groups {unknown}; known: {ALL_GROUPS}'
    seen, out = set(), []
    for it in ITEMS:
        if set(it['groups']) & set(groups) and it['key'] not in seen:
            seen.add(it['key'])
            out.append(it)
    return out


def datamap_entries():
    """The new node prefixes and their local sources (map_local when the local path has `=`)."""
    return [(it['node'], it.get('map_local', it['local'])) for it in ITEMS if it['node'].startswith(NEW_PREFIX)]


def probe_file(path):
    """One file read from an input: size and first bytes, or a numpy header. Returns a small dict."""
    import numpy as np
    size = os.path.getsize(path)
    rec = dict(file=str(path), size=size)
    if path.endswith('.npy'):
        arr = np.load(path, mmap_mode='r')
        rec.update(npy_shape=list(arr.shape), npy_dtype=str(arr.dtype))
    elif path.endswith('.npz'):
        with np.load(path) as z:
            rec.update(npz_keys=list(z.files)[:8])
    with open(path, 'rb') as f:
        rec['first_bytes'] = f.read(24).hex()
    return rec


def pick_file(path):
    """The input itself if a file, else the smallest-named regular file in the tree (npy/npz/json/csv/pt first)."""
    if os.path.isfile(path):
        return path, 1
    files = []
    for root, _, names in os.walk(path):
        files += [os.path.join(root, n) for n in names]
    assert files, f'no file under {path}'
    rank = {'.npy': 0, '.npz': 1, '.pt': 2, '.json': 3, '.csv': 4}
    files.sort(key=lambda p: (rank.get(os.path.splitext(p)[1], 9), os.path.getsize(p), p))
    return files[0], len(files)


def probe(groups):
    import importlib
    out = dict(groups=groups, tag=os.environ.get('CLUSTER_RUN_TAG', 'no-tag'), host=os.uname().nodename,
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), python=sys.version.split()[0],
               inputs=[], failures=[])
    for it in select(groups):
        rec = dict(key=it['key'], path=it['local'], groups=it['groups'])
        try:
            assert os.path.exists(it['local']), 'missing'
            f, n = pick_file(it['local'])
            rec.update(kind='dir' if os.path.isdir(it['local']) else 'file', n_files=n, **probe_file(f))
        except Exception as e:                                   # a failed input is recorded, then the probe fails
            rec['error'] = f'{type(e).__name__}: {e}'
            out['failures'].append(it['key'])
        out['inputs'].append(rec)
    out['packages'] = {}
    for name in ('numpy', 'torch', 'sklearn', 'scipy', 'threadpoolctl'):
        try:
            out['packages'][name] = importlib.import_module(name).__version__
        except Exception as e:
            out['packages'][name] = f'IMPORT FAILED {type(e).__name__}: {e}'
            out['failures'].append('import ' + name)
    try:
        import torch
        out['cuda'] = dict(available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                           devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        if not out['cuda']['available']:
            out['failures'].append('cuda not available')
    except Exception as e:
        out['cuda'] = f'FAILED {type(e).__name__}: {e}'
    results = Path(os.environ.get('BS_PROBE_OUT', '.')) / 'probe'
    results.mkdir(parents=True, exist_ok=True)
    target = results / f"probe_{out['tag']}.json"
    out['ok'] = not out['failures']
    target.write_text(json.dumps(out, indent=1))
    for rec in out['inputs']:
        print(('OK  ' if 'error' not in rec else 'FAIL'), rec['key'], rec['path'],
              rec.get('error') or f"{rec['file'].rsplit('/', 1)[-1]} {rec['size']}B {rec.get('npy_shape', '')}")
    print('packages', json.dumps(out['packages']))
    print('cuda', json.dumps(out['cuda']), 'CUDA_VISIBLE_DEVICES', out['cuda_visible_devices'])
    print(f"BS_PROBE {'ok' if out['ok'] else 'FAILED ' + str(out['failures'])} -> {target}")
    return 0 if out['ok'] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    sub.required = True                                          # the node's system python is 3.6
    g = sub.add_parser('groups-for')
    g.add_argument('mode')
    g.add_argument('--encoder')
    g.add_argument('--kind', nargs='+', default=[])
    g.add_argument('groups', nargs='*')
    for name in ('overrides', 'binds', 'verify', 'probe'):
        sub.add_parser(name).add_argument('groups', nargs='*')
    d = sub.add_parser('datamap')
    d.add_argument('--conf')
    a = ap.parse_args(argv)
    if a.cmd == 'groups-for':
        print(' '.join(groups_for(a.mode, a.encoder, a.kind, a.groups)))
    elif a.cmd == 'overrides':
        print(' '.join(f"+bs_in.{it['key']}_path={it['node']}" for it in select(a.groups)))
    elif a.cmd == 'binds':
        for it in select(a.groups):
            print(f"{it['node']}\t{it['local']}")
    elif a.cmd == 'datamap':
        have = set()
        if a.conf:
            line = next(x for x in Path(a.conf).read_text().splitlines() if x.startswith('DATA_MAP='))
            have = set(line.split('=', 1)[1].split())
        for node, local in datamap_entries():
            if f'{node}:{local}' not in have:
                print(f'{node}:{local}')
    elif a.cmd == 'verify':
        bad = []
        for it in select(a.groups or ALL_GROUPS):
            for p in (it['local'], it.get('map_local')):
                if p and not os.path.exists(p):
                    bad.append(p)
        print('missing:' if bad else 'all local sources exist', *bad, sep='\n')
        return 1 if bad else 0
    elif a.cmd == 'probe':
        return probe(a.groups)
    return 0


if __name__ == '__main__':
    sys.exit(main())
