"""Node checks of the recipe sweep (rule §4.2 "The node route"); run inside the node's fake root by
`scripts/run_rs_node.sh check|smoke`, after `das6_sync_rs.py --inputs --run`.

  n1   N1: package versions, the stores, every synced input equal to its local source by SHA-256 (the list
       `inputs_manifest.json` that the sync wrote from the local files), at least 60 GB free on the node's local disk,
       the k11 gate current for this code and the k11 graph stores present, and the round's own data loader reading
       P (P's SHA-256 = 12-25's). -> <outputs>/gates/n1.json
  n2   N2: a 2-epoch smoke of G on the GPU (rs_train --smoke) and its scoring (rs_sweep score --smoke, which holds
       V1b: 29,232 finite unit rows in P's order, P's SHA-256). -> <outputs>/gates/n2.json
Exit 0 only on pass. A reading, not a gate: the smoke's A values are recorded; nothing compares them with local.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'src/test/20261027_recipe_sweep'
sys.path.insert(0, str(ROUND))
import rs_paths  # noqa: E402

NR = Path(os.environ.get('NR', '/local/wding/r10_rs'))
OUTPUTS = rs_paths.OUTPUTS
MIN_FREE_GB = 60
K11 = (10, 30, 40)
PYTHON = os.environ.get('RS_PYTHON', sys.executable)
SMOKE_OUT = Path(os.environ.get("RS_N2_OUT", rs_paths.SMOKE_ROOT / "r10-rs-n2"))    # RS_N2_OUT, RS_DEVICE: local plumbing test only


def atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True))
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def gpu():
    import torch
    info = dict(cuda_available=torch.cuda.is_available(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if info['cuda_available']:
        info.update(count=torch.cuda.device_count(), name=torch.cuda.get_device_name(0), torch=torch.__version__,
                    torch_cuda=torch.version.cuda)
    smi = shutil.which('nvidia-smi')
    if smi:
        info['nvidia_smi'] = subprocess.run([smi, '--query-gpu=index,name,driver_version,memory.used', '--format=csv,noheader'],
                                            capture_output=True, text=True).stdout.strip().splitlines()
    return info


def n1():
    started = time.monotonic()
    expected = json.loads((NR / 'inputs_manifest.json').read_text())
    items = {}
    versions = {p: importlib.metadata.version(p) for p in expected['versions']}
    items['versions'] = dict(node=versions, local=expected['versions'], python=f'{sys.version_info.major}.{sys.version_info.minor}',
                             python_local=expected['python'],
                             **{'pass': versions == expected['versions'] and f'{sys.version_info.major}.{sys.version_info.minor}' == expected['python']})
    wrong = []
    for path, row in sorted(expected['files'].items()):
        p = Path(path)
        if not p.is_file():
            wrong.append(f'{path}: missing')
        elif p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            wrong.append(f'{path}: differs from the local file')
    items['inputs_sha256'] = dict(files=len(expected['files']), bytes=sum(r['bytes'] for r in expected['files'].values()),
                                  differ=wrong, **{'pass': not wrong})
    import numpy as np
    stores = {}
    for path, row in sorted(expected['files'].items()):
        if row['item'] == 'full_stores' and path.endswith('.npy'):
            stores[path] = list(np.load(path, mmap_mode='r').shape)
    items['qwen_stores'] = dict(shapes=stores, **{'pass': bool(stores) and all(len(s) == 2 and s[1] == rs_paths.WIDTH for s in stores.values())})
    free = shutil.disk_usage(os.environ.get('RS_DISK_PATH', '/local/wding')).free
    items['disk'] = dict(free_gb=round(free / 1e9, 1), min_gb=MIN_FREE_GB, **{'pass': free >= MIN_FREE_GB * 1e9})
    info = gpu()
    items['gpu'] = dict(info, **{'pass': bool(info['cuda_available'])})
    import rs_graphs
    try:
        rs_graphs.require_gate(OUTPUTS, 'k11_sample')
        gate = True
    except rs_graphs.GraphRefusal as error:
        gate = str(error)
    stores = {k: (rs_graphs.store_root(OUTPUTS, 'sample', k, 'aspect') / 'rs_build.json').is_file() for k in K11}
    items['k11_graphs'] = dict(gate=gate, stores=stores, **{'pass': gate is True and all(stores.values())})
    import rg_train
    data = rg_train.Data()
    _, _, p, p_sha = data.get(export_p=True)
    items['loader'] = dict(p_rows=len(p['img']), p_sha256=p_sha, **{'pass': p_sha == rs_paths.P_SHA256 and len(p['img']) == rs_paths.N_P})
    record = dict(name='n1', host=socket.gethostname(), commit=git('rev-parse', 'HEAD'), items=items,
                  inputs_manifest_sha256=sha(NR / 'inputs_manifest.json'), seconds=time.monotonic() - started,
                  **{'pass': all(i['pass'] for i in items.values())})
    atomic_json(OUTPUTS / 'gates/n1.json', record)
    for name, item in items.items():
        print(f'N1 {name}: {"PASS" if item["pass"] else "FAIL"}', {k: v for k, v in item.items() if k not in ('pass', 'shapes')}, flush=True)
    print(f'N1 {"PASS" if record["pass"] else "FAIL"} ({record["seconds"]:.0f} s) {OUTPUTS / "gates/n1.json"}', flush=True)
    return record['pass']


def n2():
    started = time.monotonic()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    train = [PYTHON, str(ROUND / 'rs_train.py'), 'train', '--configs', 'G', '--seeds', '0', '--smoke', '--device', os.environ.get('RS_DEVICE', 'cuda'),
             '--machine', 'node402', '--outputs', str(SMOKE_OUT)]
    score = [PYTHON, str(ROUND / 'rs_sweep.py'), 'score', '--machine', 'node402', '--runs', 'G', '--smoke', '--outputs', str(SMOKE_OUT)]
    codes = {}
    for label, argv in (('train', train), ('score', score)):
        print('N2', label, ' '.join(argv), flush=True)
        codes[label] = subprocess.run(argv, cwd=ROOT).returncode
        if codes[label]:
            break
    items = dict(exit_codes=codes, gpu=gpu())
    ok = codes == {'train': 0, 'score': 0}
    if ok:
        run = json.loads((rs_paths.run_dir(SMOKE_OUT, 'G', 0) / 'run.json').read_text())
        scored = json.loads((SMOKE_OUT / 'scores/node402/G__s0.json').read_text())
        items.update(run_status=run['status'], run_machine=run['machine'], epochs_run=run['epochs_run'], score_status=scored['status'],
                     score_host=scored['host'], A=scored['A'], img_sha256=scored['img_sha256'], p_sha256=scored['p_sha256'])
        ok = (run['status'] == 'ok' and run['machine'] == 'node402' and run['smoke'] is True and scored['status'] == 'ok'
              and scored['p_sha256'] == rs_paths.P_SHA256 and scored['smoke'] is True and set(scored['A']) == set(rs_paths.ASPECTS)
              and all(0 <= v <= 1 for v in scored['A'].values()))
    record = dict(name='n2', host=socket.gethostname(), commit=git('rev-parse', 'HEAD'), items=items,
                  seconds=time.monotonic() - started, **{'pass': bool(ok)})
    atomic_json(OUTPUTS / 'gates/n2.json', record)
    print(f'N2 {"PASS" if ok else "FAIL"} ({record["seconds"]:.0f} s) A={items.get("A")} {OUTPUTS / "gates/n2.json"}', flush=True)
    return ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('check', choices=['n1', 'n2'])
    raise SystemExit(0 if {'n1': n1, 'n2': n2}[parser.parse_args().check]() else 1)
