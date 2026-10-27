#!/usr/bin/python3
"""Input plumbing of the r10 recipe sweep's node route (rule §4.2 "The node route", spec D19), modelled on
`das6_sync_ct2.py`. Everything goes through the cluster skill's `cluster` module. No node contact unless --run (or
--verify, --fetch with --run) is given; --inputs is the default and a dry run prints the plan and exits 3 if a local
source is missing.

  --inputs [--run]       copy every item of rs_node_inputs.json to NODE:/local/wding/r10_rs/<remote>, then write
                         NODE:/local/wding/r10_rs/inputs_manifest.json (SHA-256 of every file, as the local source has it
                         at this moment; the package versions of the local env). N1 reads it.
  --verify               compare the node's files and bytes with the local sources (no hashes: N1 does those)
  --fetch REL [REL ...]  [--run] copy node outputs/REL home, refusing to overwrite a differing local file. REL under
                         scores/node402, sweep/node402 or gates (n1, n2, n3) lands at the same path in the local outputs;
                         any other REL (train/<run>, full/...) lands under <outputs>/node402/REL, since the local runs
                         use the same names.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

SKILL = Path.home() / '.claude/skills/cluster-run'
sys.path.insert(0, str(SKILL))
import cluster  # noqa: E402

NR = '/local/wding/r10_rs'                  # INPUTS_REMOTE of this route (add it to projects/CoSiR.conf for `cluster du`)
MANIFEST = Path(__file__).with_name('rs_node_inputs.json')
LOCAL_OUTPUTS = Path(os.environ.get('RS_OUTPUTS', '/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs'))
# Copy of das6_sync_ct2.EXCLUDED: GPU-serving packages of the local env the sweep never imports (about 10 GB).
EXCLUDED = 'libcugraph libcuvs flashinfer_cubin libcudf vllm libcuml libkvikio cupy PyQt5'.split()
VERSION_PACKAGES = ('torch', 'numpy', 'scipy', 'scikit-learn', 'threadpoolctl')
FETCH_SAME_PATH = ('scores/node402', 'sweep/node402', 'gates/n1.json', 'gates/n2.json', 'gates/n3.json',
                   'gates/n3_scorer_parity_partitions')


def remote_path(relative):
    p = Path(NR) / relative
    if not str(p).startswith(NR + '/') or '..' in p.parts:
        raise ValueError(f'remote path outside {NR}: {p}')
    return str(p)


def items():
    result = json.loads(MANIFEST.read_text())
    assert len({i['key'] for i in result}) == len(result)
    return result


def kind(item):
    if 'files' in item:
        return 'selected'
    return 'file' if Path(item['local']).is_file() else 'dir'


def files(item):
    """{relative name: bytes} of what the item copies (symlinks dereferenced, as rsync -L; the env's exclusions)."""
    base = Path(item['local'])
    if not base.exists():
        raise FileNotFoundError(base)
    if 'files' in item:
        return {name: (base / name).stat().st_size for name in item['files']}     # FileNotFoundError if absent
    if base.is_file():
        return {base.name: base.stat().st_size}
    found = {}
    for folder, dirs, names in os.walk(base, followlinks=True):
        rel = Path(folder).relative_to(base)
        dirs[:] = [d for d in dirs if not (item.get('exclude_env') and rel.name == 'site-packages' and d in EXCLUDED)]
        for name in names:
            p = Path(folder) / name
            if not p.is_file():
                raise ValueError(f'not a readable regular file: {p}')
            found[str(p.relative_to(base))] = p.stat().st_size
    return found


def exclusions(item):
    return ['lib/python*/site-packages/' + x + '/***' for x in EXCLUDED] if item.get('exclude_env') else []


def sync(node, item):
    action = dict(item, kind=kind(item), remote=remote_path(item['remote']))
    target = action['remote'] if action['kind'] != 'file' else str(Path(action['remote']).parent)
    cluster.require_remote(cluster.remote(node, 'mkdir -p -- "$DIR"', {'DIR': target}, 30), 'mkdir input')
    listing = None
    try:
        if action['kind'] == 'selected':
            with tempfile.NamedTemporaryFile('w', suffix='.files', delete=False) as handle:
                handle.write('\n'.join(item['files']) + '\n')
            listing = handle.name
        argv = cluster.rsync_argv(node, action, listing)
        # Dereference env symlinks: external targets must not escape the copied tree.
        argv[2:2] = ['--copy-links', *['--exclude=' + x for x in exclusions(item)]]
        cluster.require_local(argv, 'sync ' + item['key'], 6 * 3600)
    finally:
        if listing:
            os.unlink(listing)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def input_manifest():
    """What N1 compares the node with: SHA-256 and bytes of every file but the env's, keyed by the absolute path the
    code sees (every item is bound at its local path), and the local env's package versions."""
    table = {}
    for item in items():
        if item['key'] == 'env':
            continue
        for name, size in files(item).items():
            path = Path(item['local']) if kind(item) == 'file' else Path(item['local']) / name
            table[str(path)] = dict(sha256=sha(path), bytes=size, item=item['key'])
    versions = {p: importlib.metadata.version(p) for p in VERSION_PACKAGES}
    return dict(files=table, versions=versions, python=f'{sys.version_info.major}.{sys.version_info.minor}')


def write_manifest(node):
    record = input_manifest()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'inputs_manifest.json'
        path.write_text(json.dumps(record, indent=1, sort_keys=True))
        cluster.require_local(cluster.rsync_argv(node, dict(kind='file', local=str(path),
                                                            remote=remote_path('inputs_manifest.json'))),
                              'sync inputs_manifest', 600)
    print(f'inputs_manifest.json: {len(record["files"])} files hashed, versions {record["versions"]}')


# Node inventory through cluster.remote only. Bytes and, when asked, SHA-256 of regular files under a path.
INVENTORY = r'''
import hashlib,json,os
from pathlib import Path
p=Path(os.environ['PATH_RS']); hashes=os.environ.get('HASH_RS')=='1'
if not p.exists(): raise SystemExit('missing input: '+str(p))
items=[p] if p.is_file() else sorted(p.rglob('*'))
result={}
for f in items:
    if hashes and f.is_symlink(): raise SystemExit('refusing fetch symlink: '+str(f))
    if f.is_file():
        key=f.name if p.is_file() else str(f.relative_to(p))
        row={'bytes':f.stat().st_size}
        if hashes:
            h=hashlib.sha256()
            with f.open('rb') as stream:
                for b in iter(lambda:stream.read(1024*1024),b''): h.update(b)
            row['sha256']=h.hexdigest()
        result[key]=row
print(json.dumps(result))
'''


def inventory(node, path, hashes=False):
    assert path.startswith(NR + '/') and '..' not in Path(path).parts
    command = '/usr/bin/python3 -c ' + shlex.quote(INVENTORY)
    return json.loads(cluster.require_remote(cluster.remote(
        node, command, {'PATH_RS': path, 'HASH_RS': '1' if hashes else '0'}, 900, retry=False), 'inventory'))


def included(item, name):
    parts = Path(name).parts
    if item.get('exclude_env'):
        for i, part in enumerate(parts[:-1]):
            if part == 'site-packages' and parts[i + 1] in EXCLUDED:
                return False
    return True


def verify(node, item):
    expected = files(item)
    actual = inventory(node, remote_path(item['remote']))
    if 'files' in item or item.get('outputs'):
        # the node's folder may hold extra files (the node outputs are written there): compare the listed ones
        got = {n: actual[n]['bytes'] for n in expected if n in actual}
        ok = got == expected
        print(item['key'], f'node has {len(got)} of {len(expected)} listed files', 'bytes equal' if ok else 'DIFFER')
    else:
        actual = {n: r['bytes'] for n, r in actual.items() if included(item, n)}
        ok = actual == expected
        print(item['key'], 'node', (len(actual), sum(actual.values())), 'local', (len(expected), sum(expected.values())))
    if not ok:
        raise SystemExit('VERIFY FAILED: ' + item['key'])


def fetch_target(rel):
    assert not Path(rel).is_absolute() and '..' not in Path(rel).parts, rel
    same = any(rel == x or rel.startswith(x + '/') for x in FETCH_SAME_PATH)
    return LOCAL_OUTPUTS / (rel if same else f'node402/{rel}')


def fetch(node, rel, run):
    """Copy node outputs/REL home. A differing local file refuses the whole fetch; a local file is never overwritten."""
    source, local = remote_path('outputs/' + rel), fetch_target(rel)
    print(f'Fetch plan: {node}:{source} -> {local} (refuse conflicts)')
    if not run:
        return
    before = inventory(node, source, True)
    single = len(before) == 1 and next(iter(before)) == Path(source).name and not rel.endswith('/')
    conflicts = []
    for name, row in before.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts, name
        p = local.parent / name if single else local / name
        if any(q.is_symlink() or (q.exists() and not q.is_dir()) for q in p.parents):
            conflicts.append(name + ' (unsafe parent)')
        elif p.is_symlink() or (p.exists() and (not p.is_file() or sha(p) != row['sha256'])):
            conflicts.append(name)
    if conflicts:
        raise SystemExit('REFUSING: differing local files:\n' + '\n'.join(conflicts))
    local.parent.mkdir(parents=True, exist_ok=True)
    if single:
        argv = cluster.rsync_argv(node, dict(kind='file', local=str(local), remote=source))
    else:
        local.mkdir(parents=True, exist_ok=True)
        argv = cluster.rsync_argv(node, dict(kind='dir', local=str(local), remote=source))
    argv[-2:] = [argv[-1], argv[-2]]
    argv[2:2] = ['--ignore-existing']          # never overwrite a local file, including one created since the scan
    cluster.require_local(argv, 'fetch ' + rel, 6 * 3600)
    after = inventory(node, source, True)
    for name, row in after.items():
        p = local.parent / name if single else local / name
        if not p.is_file() or sha(p) != row['sha256']:
            raise SystemExit(f'Fetch not verified: {p}')
    if before != after:
        raise SystemExit('Fetch not verified: the node source changed during the copy; nothing deleted.')
    print(f'Fetch verified: {rel}: {len(after)} files')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--node', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inputs', action='store_true', help='default')
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--fetch', nargs='+', metavar='REL')
    parser.add_argument('--run', action='store_true', help='contact the node (--inputs, --fetch); --verify always does')
    args = parser.parse_args()
    cfg = cluster.load_config(SKILL / 'cluster.conf', project='CoSiR')
    node = cluster.validate_node(args.node, cfg)
    if args.fetch:
        for rel in args.fetch:
            fetch(node, rel, args.run)
        return 0
    total, missing = 0, []
    plan = []
    for item in items():
        try:
            listing = files(item)
        except FileNotFoundError as error:
            missing.append(f'{item["key"]}: {error}')
            print(f'{item["key"]}: MISSING locally: {error}')
            continue
        total += sum(listing.values())
        plan.append(item)
        print(f'{item["key"]}: {len(listing)} files {sum(listing.values()):,} bytes ({sum(listing.values())/1024**3:.3f} GiB)'
              f' -> {remote_path(item["remote"])}')
    print(f'Total: {total:,} bytes; {total/1024**3:.3f} GiB')
    if missing:
        print('Build the missing sources first (the k11 graph variants: rs_graphs build); nothing is copied.')
        return 3
    if args.verify:
        for item in plan:
            verify(node, item)
        print('VERIFY: all items equal in files and bytes')
        return 0
    if args.run:
        for item in plan:
            print('sync', item['key'], flush=True)
            sync(node, item)
        write_manifest(node)
    else:
        print('Local plan only; --run enables node contact. No deletions.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
