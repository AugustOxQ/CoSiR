"""Read-only TPIPS shard audit: compare the 48 MMArt probe rows with the batch-1 reference, per shard and factor.

Writes only OUT/stage0/tpips_store_check.json (a run report, never a status gate), exits 1 on a failed comparison or
*.dist-info newer than the first shard's start. No model, extraction, merge or GPU is run. Uses gate (iv)'s TPIPS_COS.
Finished shard files or chunks older than 120 seconds are read; active/unseen probe rows are reported as pending,
not failed. A finished shard must contain every probe row assigned by the MMArt length-sorted input manifest.

CLI: [--stores SET=DIR ...] [--probe DIR] [--manifest MMART_JSON] [--run-log FILE ...] [--started-at ISO_TIME].
Stores default to om_paths.TPIPS_STORE (even with OM_OUTPUTS); probe defaults to OUT/stage0/probe_b1. The manifest and
run logs default to DEFAULT_OUT/tpips_inputs/mmart.json and DEFAULT_OUT/tpips_local_*.log, so a scratch OM_OUTPUTS can
audit the live stores via --probe. Start lines use Europe/Amsterdam; --started-at needs an explicit timezone.
If no start log is available, the earliest shard directory birth time is used and identified as a start proxy.
Package directories are scanned in the CoSiR environment and /root/venvs/tpips, including their lib/python*/site-packages.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

import om_paths as P
import om_tpips as tp

ENVIRONMENTS = (Path('/root/miniconda3/envs/CoSiR'), P.TPIPS_ENV)
STABLE_SECONDS = 120


def read(path):
    return json.loads(Path(path).read_text())


def shard_start(stores, logs, explicit=None):
    """Return the first logged shard start (minute precision, conservative), else the earliest directory birth."""
    if explicit:
        parsed = datetime.fromisoformat(explicit)
        if parsed.tzinfo is None:
            raise ValueError('--started-at needs a timezone')
        return parsed.timestamp(), '--started-at'
    starts = []
    for path in logs:
        for line in Path(path).read_text().splitlines():
            hit = re.fullmatch(r'shard \d+ start (\d{4}-\d\d-\d\d \d\d:\d\d)', line)
            if hit:
                dt = datetime.strptime(hit[1], '%Y-%m-%d %H:%M').replace(tzinfo=ZoneInfo('Europe/Amsterdam'))
                starts.append((dt.timestamp(), f'{path}: {line}'))
    if starts:
        return min(starts)
    for root in stores.values():
        for path in Path(root).glob('*/shards/*'):
            if path.is_dir():
                result = subprocess.run(['stat', '-c', '%W', str(path)], capture_output=True, text=True, check=True)
                birth = int(result.stdout.strip())
                if birth > 0:
                    starts.append((birth, f'directory birth (start proxy): {path}'))
    if not starts:
        raise ValueError('no shard start available; pass --run-log or --started-at with a timezone')
    return min(starts)


def package_changes(environments, started):
    """List dist-info directories newer than start, independently in both environments (no Python subprocess)."""
    result = {}
    for env in environments:
        changed = []
        for path in sorted(Path(env).glob('lib/python*/site-packages/*.dist-info')):
            modified = path.stat().st_mtime
            if modified > started:
                changed.append(dict(path=str(path), mtime=modified,
                                    modified_at=datetime.fromtimestamp(modified, ZoneInfo('UTC')).isoformat()))
        result[str(env)] = changed
    return result


def _probe(probe):
    refs, keys = {}, None
    for factor in tp.FACTORS:
        root = Path(probe) / factor
        meta, ks = read(root / 'meta.json'), read(root / 'keys.json')
        assert len(ks) == len(set(ks)) == P.TPIPS_PROBE_N, f'{factor}: expected {P.TPIPS_PROBE_N} unique probe keys'
        assert meta.get('batches') == [1] and not meta.get('fake'), f'{factor}: reference is not real batch 1'
        assert keys is None or set(ks) == keys, f'{factor}: reference factors have different keys'
        keys = set(ks)
        assert meta['sha256'] == P.sha256_file(root / 'emb.npy'), f'{factor}: reference sha256 differs'
        x = np.load(root / 'emb.npy', mmap_mode='r')
        assert x.shape == (len(ks), tp.WIDTH), f'{factor}: reference shape {x.shape}'
        refs[factor] = {k: np.asarray(x[i], dtype=np.float64) for i, k in enumerate(ks)}
    return refs, keys


def compare_shard(path, reference, expected, now):
    """Read stable chunks or the finished shard, retaining only probe rows; incomplete shards can be pending."""
    path = Path(path)
    finished = (path / 'meta.json').exists()
    row = dict(path=str(path), finished=finished, expected=len(expected), n=0, min_cos=None, per_key={},
               pending_keys=[], active_chunks=[], problems=[])
    if finished:
        sources = [(path / 'emb.npy', path / 'keys.json')]
    else:
        sources = []
        for p in sorted((path / 'chunks').glob('*.npy')):
            if now - p.stat().st_mtime < STABLE_SECONDS or not p.with_suffix('.json').exists():
                row['active_chunks'].append(p.name)
            else:
                sources.append((p, p.with_suffix('.json')))
    for p, key_path in sources:
        try:
            before = p.stat()
            ks = read(key_path)
            hits = [(i, k) for i, k in enumerate(ks) if k in reference]
            if not hits:
                continue
            x = np.load(p, mmap_mode='r')
            assert x.shape == (len(ks), tp.WIDTH), f'{p}: keys/shape differ: {x.shape}'
            values = {}
            for i, key in hits:
                a, b = np.asarray(x[i], dtype=np.float64), reference[key]
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                assert np.isfinite(a).all() and np.isfinite(b).all() and norm > 0, f'{p}: invalid row {key}'
                values[key] = float(np.dot(a, b) / norm)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                row['active_chunks'].append(p.name)
                continue
            for key, cosine in values.items():
                assert key not in row['per_key'], f'{p}: duplicate probe row {key}'
                row['per_key'][key] = cosine
        except (OSError, ValueError, AssertionError, KeyError) as exc:
            row['problems'].append(str(exc))
    found = set(row['per_key'])
    row['pending_keys'] = sorted(expected - found)
    if found - expected:
        row['problems'].append(f'probe keys assigned to another shard: {sorted(found - expected)}')
    if finished and row['pending_keys']:
        row['problems'].append(f'finished shard lacks {len(row["pending_keys"])} expected probe rows')
    row['n'] = len(found)
    if found:
        row['min_cos'] = min(row['per_key'].values())
        if row['min_cos'] < P.TPIPS_COS:
            row['problems'].append(f'lowest cosine {row["min_cos"]} < {P.TPIPS_COS}')
    row['pass'] = not row['problems']
    row['state'] = 'failed' if not row['pass'] else 'pending' if not finished or row['pending_keys'] else 'complete'
    return row


def check(stores, probe, manifest, logs, started_at=None, environments=ENVIRONMENTS, now=None):
    """Build the report without writing; missing/active shards remain visible and do not fabricate a comparison."""
    report = dict(stores={k: str(v) for k, v in stores.items()}, probe=str(probe), manifest=str(manifest),
                  cos_min=P.TPIPS_COS, rows=[], problems=[], package_changes={}, probe_rows_by_shard={},
                  pending_shards=[])
    try:
        started, source = shard_start(stores, logs, started_at)
        report.update(started=started, started_at=datetime.fromtimestamp(started, ZoneInfo('UTC')).isoformat(),
                      start_source=source)
        report['package_changes'] = package_changes(environments, started)
        for env, changes in report['package_changes'].items():
            if changes:
                report['problems'].append(f'{env}: {len(changes)} dist-info directories newer than shard start')
        refs, probe_keys = _probe(probe)
        report['probe_n'] = len(probe_keys)
        man = read(manifest)
        assert man['set'] == 'mmart' and probe_keys <= set(man['keys']), 'probe keys are not all MMArt test keys'
        assert sorted(man['order']) == list(range(len(man['keys']))), 'invalid manifest order'
        for set_, root in stores.items():
            for factor in tp.FACTORS:
                for path in sorted((Path(root) / factor / 'shards').glob('*')):
                    if not path.is_dir():
                        continue
                    match = re.fullmatch(r'(\d+)of(\d+)', path.name)
                    assert match, f'invalid shard name {path}'
                    i, n = map(int, match.groups())
                    if set_ == 'mmart':
                        for shard in range(n):
                            tag = f'{shard}of{n}'
                            assigned = probe_keys & {man['keys'][j] for j in tp.shard_indices(man, shard, n)}
                            report['probe_rows_by_shard'][tag] = len(assigned)
                    expected = (probe_keys & {man['keys'][j] for j in tp.shard_indices(man, i, n)}
                                if set_ == 'mmart' else set())
                    row = compare_shard(path, refs[factor], expected, time.time() if now is None else now)
                    report['rows'].append(dict(set=set_, factor=factor, shard=path.name, **row))
                    report['problems'].extend(f'{set_}/{factor}/{path.name}: {p}' for p in row['problems'])
        if not report['rows']:
            report['problems'].append('no shard directories exist')
        seen = {(r['set'], r['factor'], r['shard']) for r in report['rows']}
        report['pending_shards'] = [f'mmart/{factor}/{tag}' for factor in tp.FACTORS
                                    for tag in report['probe_rows_by_shard'] if ('mmart', factor, tag) not in seen]
    except (OSError, ValueError, AssertionError, KeyError, subprocess.SubprocessError) as exc:
        report['problems'].append(f'{type(exc).__name__}: {exc}')
    report['pass'] = not report['problems']
    report['complete'] = bool(report['rows']) and not report['pending_shards'] and all(r['state'] == 'complete'
                                                                                 for r in report['rows'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--stores', action='append', metavar='SET=DIR', help='repeat for artelingo and/or mmart')
    parser.add_argument('--probe', type=Path, default=P.OUT / 'stage0' / 'probe_b1')
    parser.add_argument('--manifest', type=Path, default=P.DEFAULT_OUT / 'tpips_inputs' / 'mmart.json')
    parser.add_argument('--run-log', action='append', type=Path)
    parser.add_argument('--started-at', help='ISO time with timezone; otherwise read shard start lines from run logs')
    args = parser.parse_args(argv)
    stores = dict(P.TPIPS_STORE) if not args.stores else dict(s.split('=', 1) for s in args.stores)
    if not set(stores) <= {'mmart', 'artelingo'}:
        parser.error('--stores needs mmart=DIR or artelingo=DIR')
    logs = args.run_log if args.run_log is not None else sorted(P.DEFAULT_OUT.glob('tpips_local_*.log'))
    report = check(stores, args.probe, args.manifest, logs, args.started_at)
    target = P.OUT / 'stage0' / 'tpips_store_check.json'
    P.write_json(target, report)
    for row in report['rows']:
        print(f'{row["set"]}/{row["shard"]}/{row["factor"]}: {row["n"]}/{row["expected"]} probe rows, '
              f'lowest cosine {row["min_cos"]}, {row["state"]}')
    for env, changes in report['package_changes'].items():
        print(f'{env}: {len(changes)} newer dist-info directories')
        for change in changes:
            print(f'  {change["path"]}: {change["modified_at"]}')
    for problem in report['problems']:
        print(f'FAIL: {problem}')
    print(f'{"PASS" if report["pass"] else "FAIL"} (complete={report["complete"]}) -> {target}')
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
