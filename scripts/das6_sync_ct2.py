#!/usr/bin/python3
"""CT2 input plumbing via cluster.py; no node contact unless --run is supplied."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

SKILL = Path.home() / '.claude/skills/cluster-run'
sys.path.insert(0, str(SKILL))
import cluster

NR = '/local/wding/r10_ct2'
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name('ct2_node_inputs.json')
EXCLUDED = 'libcugraph libcuvs flashinfer_cubin libcudf vllm libcuml libkvikio cupy PyQt5'.split()


def remote_path(relative):
    p = Path(NR) / relative
    if not str(p).startswith(NR + '/') or '..' in p.parts:
        raise ValueError(f'remote path outside {NR}: {p}')
    return str(p)


def exclusions(item):
    result = ['/' + x + '/***' for x in item.get('exclude', [])]
    if item.get('exclude_env'):
        result += ['lib/python*/site-packages/' + x + '/***' for x in EXCLUDED]
    return result


def files(item):
    """Regular-file names and dereferenced bytes; match rsync -aL and exclusions."""
    base = Path(item['local'])
    if not base.exists():
        raise FileNotFoundError(base)
    if base.is_file():
        return {base.name: base.stat().st_size}
    found = {}
    for folder, dirs, names in os.walk(base, followlinks=True):
        rel = Path(folder).relative_to(base)
        dirs[:] = [d for d in dirs if not
                   (rel == Path('.') and d in item.get('exclude', [])) and not
                   (item.get('exclude_env') and rel.name == 'site-packages' and d in EXCLUDED)]
        for name in names:
            p = Path(folder) / name
            if not p.is_file():
                raise ValueError(f'not a readable regular file: {p}')
            found[str(p.relative_to(base))] = p.stat().st_size
    return found


def actions():
    result = []
    for item in json.loads(MANIFEST.read_text()):
        if item.get('controller'):
            continue
        listing = files(item)
        result.append(dict(item, kind='file' if Path(item['local']).is_file() else 'dir',
                           remote=remote_path(item['remote']), files_count=len(listing),
                           bytes=sum(listing.values())))
    return result


def sync(node, item):
    target = item['remote'] if item['kind'] == 'dir' else str(Path(item['remote']).parent)
    cluster.require_remote(cluster.remote(node, 'mkdir -p -- "$DIR"', {'DIR': target}, 30), 'mkdir input')
    argv = cluster.rsync_argv(node, item)
    # Dereference env/model symlinks: external targets must not escape the copied tree.
    argv[2:2] = ['--copy-links', *['--exclude=' + x for x in exclusions(item)]]
    cluster.require_local(argv, 'sync ' + item['key'], 6 * 3600)


# Node inventory uses cluster.remote, never a direct SSH invocation. Includes hashes
# for fetch conflict checks; regular files only, with all symlinks refused on fetch.
INVENTORY = r'''
import hashlib,json,os
from pathlib import Path
p=Path(os.environ['PATH_CT2']); hashes=os.environ.get('HASH_CT2')=='1'
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
    # All remote inventory paths are subject to the same prefix assertion.
    assert path.startswith(NR + '/') and '..' not in Path(path).parts
    import shlex
    command = '/usr/bin/python3 -c ' + shlex.quote(INVENTORY)
    return json.loads(cluster.require_remote(cluster.remote(
        node, command, {'PATH_CT2': path, 'HASH_CT2': '1' if hashes else '0'}, 240,
        retry=False), 'inventory'))


def included(item, name):
    parts = Path(name).parts
    if parts[0] in item.get('exclude', []):
        return False
    if item.get('exclude_env'):
        for i, part in enumerate(parts[:-1]):
            if part == 'site-packages' and parts[i + 1] in EXCLUDED:
                return False
    return True


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fetch(node):
    source = remote_path('outputs/stage2')
    local = ROOT / 'src/test/20261020_claim_test/outputs/stage2'
    before = inventory(node, source, True)
    conflicts = []
    for name, row in before.items():
        p = local / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'unsafe fetch path: {name}')
        if any(q.is_symlink() or (q.exists() and not q.is_dir()) for q in p.parents):
            conflicts.append(name + ' (unsafe parent)')
        elif p.is_symlink() or (p.exists() and (not p.is_file() or sha(p) != row['sha256'])):
            conflicts.append(name)
    if conflicts:
        raise SystemExit('REFUSING: differing local files:\n' + '\n'.join(conflicts))
    local.mkdir(parents=True, exist_ok=True)
    action = dict(kind='dir', local=str(local), remote=source)
    argv = cluster.rsync_argv(node, action)
    argv[-2:] = [argv[-1], argv[-2]]
    # Never overwrite a local file, including one created since the conflict scan.
    argv[2:2] = ['--ignore-existing']
    cluster.require_local(argv, 'fetch stage2', 6 * 3600)
    after = inventory(node, source, True)
    if before != after or any(not (local / n).is_file() or sha(local / n) != r['sha256']
                              for n, r in after.items()):
        raise SystemExit('Fetch not verified: source changed or local file differs; nothing deleted.')
    print(f'Fetch verified: {len(after)} files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('inputs', 'fetch', 'verify'):
        mode.add_argument('--' + flag, action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--with-possible', action='store_true',
                        help='also copy or verify conditional extraction inputs')
    args = parser.parse_args()
    cfg = cluster.load_config(SKILL / 'cluster.conf', project='CoSiR')
    node = cluster.validate_node(args.node, cfg)
    if args.fetch:
        print(f'Fetch plan: {node}:{remote_path("outputs/stage2")} -> '
              f'{ROOT}/src/test/20261020_claim_test/outputs/stage2 (refuse conflicts)')
        if args.run:
            fetch(node)
    else:
        items = []
        for a in actions():
            if a['certainty'] == 'possible' and not args.with_possible:
                print(f'{a["key"]}: skipped (possible), {a["bytes"]:,} bytes '
                      f'({a["bytes"]/1024**3:.3f} GiB)')
                continue
            items.append(a)
            print(f'{a["key"]}: {a["files_count"]} files {a["bytes"]:,} bytes '
                  f'({a["bytes"]/1024**3:.3f} GiB) -> {a["remote"]}')
        print(f'Total: {sum(a["bytes"] for a in items):,} bytes; '
              f'{sum(a["bytes"] for a in items)/1024**3:.3f} GiB')
        if args.run:
            for a in items:
                if args.inputs:
                    sync(node, a)
                else:
                    actual = {name: row for name, row in inventory(node, a['remote']).items()
                              if included(a, name)}
                    expected = files(a)
                    got = (len(actual), sum(r['bytes'] for r in actual.values()))
                    want = (len(expected), sum(expected.values()))
                    print(a['key'], 'node', got, 'local', want)
                    if got != want:
                        raise SystemExit('VERIFY FAILED: ' + a['key'])
    if not args.run:
        print('Local plan only; --run enables node contact. No deletions.')


if __name__ == '__main__':
    main()
